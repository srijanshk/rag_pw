from __future__ import annotations

import logging
import json, os, re, random
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import List, Optional, Dict

import fire, torch
from datasets import load_dataset
from rich.progress import track

import faiss 
import numpy as np

from FlagEmbedding import BGEM3FlagModel
from BGERetriever import BGERetriever
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
import wandb 

# ---------------------------------------------------------------------------
# Llama‑3.1 prompt
# ---------------------------------------------------------------------------
# SYSTEM_MSG = (
#         "You are an expert mathematician. Combine the retrieved information below "
#         "with your own knowledge to solve the problem. Use the context mainly as "
#         "examples or hints—you may add any facts you already know. "
#         "Think step-by-step inside <think> … </think> and finish with exactly one "
#         "line <answer> … </answer> containing only the numeric answer."
#     )
SYSTEM_MSG = """
    You are an expert mathematician.
    Combine the retrieved information below with your own knowledge to solve the problem.
    Use the context mainly as examples or hints—you may add any facts you already know.

    Think step-by-step.  
    Write every reasoning step inside `<think> … </think>` blocks.  
    When you are completely done, produce **exactly one**
    <answer> FINAL_NUMERIC_ANSWER </answer>

    Note: Nothing after `</answer>`.
    """

CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n{system}\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Here are a few worked-out examples you may find helpful:\n\n"
    "{retrieved_context}\n\n"
    "Now solve the following problem:\n{question}\n<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------
CLOSED_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
OPEN_RE   = re.compile(r"<answer>(.*)$", re.S | re.I)
PRE_CLOSE_RE = re.compile( 
    r"([+-]?\d+(?:/\d+)?)(?:\s*|\s*</answer>)",
    re.S | re.I,
)
NUM_RE = re.compile(
    r"(?:"
    r"\\boxed{([^}]*)}"                        # \boxed{42} or \boxed{\dfrac{…}}
    r"|\\(?:d)?frac{([^}]*)}{([^}]*)}"         # \frac{a}{b} or \dfrac{a}{b}
    r"|[+-]?\\d+(?:/\\d+)?"                    # plain 123 or 9/7
    r")",
    re.S | re.I,
)

def _clean(num: str):
    """Strip currency, commas, and LaTeX wrappers; normalise frac → a/b."""
    if num is None:
        return None
    num = num.strip()
    num = re.sub(r"\\boxed{([^}]*)}", r"\\1", num)
    num = re.sub(r"\\(?:d)?frac{([^}]*)}{([^}]*)}", r"\\1/\\2", num)
    return num.lstrip("$€£ ").replace(",", "").strip()


def strip_answer(txt: str):
    """Extract final numeric answer from assistant text, robust to LaTeX."""
    if not txt:
        return None
    # pre-flatten LaTeX
    txt_flat = re.sub(r"\\boxed{([^}]*)}", r"\\1", txt)
    txt_flat = re.sub(r"\\(?:d)?frac{([^}]*)}{([^}]*)}", r"\\1/\\2", txt_flat)

    # 1) closed tag
    if (m := CLOSED_RE.findall(txt_flat)):
        return _clean(m[-1])

    # 2) open tag tail
    if (m := OPEN_RE.findall(txt_flat)):
        tail_nums = NUM_RE.findall(m[-1])
        flat = [p for g in tail_nums for p in (g if isinstance(g, tuple) else [g]) if p]
        if flat:
            return _clean(flat[-1])
    # NEW ─ capture number just before </answer>
    if (m := PRE_CLOSE_RE.findall(txt)):
        return _clean(m[-1])

    # 3) fallback – last numeric chunk anywhere
    nums = NUM_RE.findall(txt_flat)
    flat = [p for g in nums for p in (g if isinstance(g, tuple) else [g]) if p]
    return _clean(flat[-1]) if flat else None


# ---------------------------------------------------------------------------
# Custom stopping criterion
# ---------------------------------------------------------------------------
class StopAfterAnswer(StoppingCriteria):
    """
    Stop when either
    1. the exact `</answer>` is produced, or
    2. an `<answer>` tag that already contains a digit is followed by
       (a) a newline  OR
       (b) any whitespace right after the digit  – covers cases like
          "<answer> 3 </" where the model forgets the closing tag.
    """
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.close_ids = tokenizer("</answer>").input_ids[1:]
        self.open_ids  = tokenizer("<answer>").input_ids[1:]
        self.nl_id     = tokenizer("\n").input_ids[1]
        self.sp_id     = tokenizer(" ").input_ids[1]

    def _match_tail(self, ids, pattern):
        return ids.shape[1] >= len(pattern) and torch.all(
            ids[0, -len(pattern):] == torch.tensor(pattern, device=ids.device)
        )

    def _answer_has_digit(self, ids) -> bool:
        tail = ids[0, -256:].tolist()
        text = self.tok.decode(tail, skip_special_tokens=True)
        return bool(re.search(r"<answer>[^\\n]*\\d", text, re.S))

    def __call__(self, input_ids, scores, **kwargs):
        # 1) closed tag seen
        if self._match_tail(input_ids, self.close_ids):
            return True
        # 2a) "<answer>...\\n" with a digit
        if (self._match_tail(input_ids, [*self.open_ids, self.nl_id])
                and self._answer_has_digit(input_ids)):
            return True
        # 2b) "<answer> number  " and we just emitted a space
        if (self._match_tail(input_ids, [self.sp_id])
                and self._answer_has_digit(input_ids)):
            return True
        return False

# ---------------------------------------------------------------------------
# Dataset loader (handles GSM8K & MATH JSONL)
# ---------------------------------------------------------------------------

def load_split(path: str, split: str):
    # Support common aliases for HF datasets
    alias = path.lower()
    if alias in {"math500", "math-500"}:
        path = "HuggingFaceH4/MATH-500"
    elif alias in {"math", "hendrycks/math", "hendrycks/competition_math"}:
        path = "hendrycks/competition_math"
    elif alias in {"gsm", "gsm8k"}:
        path = "gsm8k"

    if os.path.exists(path):
        if path.endswith((".json", ".jsonl")):
            return load_dataset("json", data_files={split: path}, split=split)
        if os.path.isdir(path):
            fp = os.path.join(path, f"{split}.jsonl")
            if os.path.isfile(fp):
                return load_dataset("json", data_files={split: fp}, split=split)
            import glob
            files = glob.glob(os.path.join(path, "**", f"{split}.jsonl"), recursive=True)
            if files:
                return load_dataset("json", data_files={split: files}, split=split)
        return load_dataset(path, split=split)
    return load_dataset(path, split=split)

def colbert_scores_safe(
    query: str,
    docs: list[dict],
    model: BGEM3FlagModel,
    batch_pairs: int = 16,
) -> list[float]:
    """
    Compute ColBERT scores for `query` vs each doc['solution_chunk'] (or 'text').

    • Splits work into ≤ `batch_pairs` pairs per call to keep CUDA memory low.
    • Truncates passages to `batch_tokens` tokens.
    • If einsum OOM / shape error occurs, falls back to the dense score already in doc.
    """
    pairs = [[query, d.get("solution_chunk") or d.get("text", "")] for d in docs]
    scores: list[float] = []

    for i in range(0, len(pairs), batch_pairs):
        chunk = pairs[i:i + batch_pairs]
        try:
            s = model.compute_score_single_device(
                chunk,
                batch_size=batch_pairs,
                max_query_length=128,
            )["colbert"]
        except (RuntimeError, ValueError) as e:
            logging.warning(
                "ColBERT scorer failed on docs %d–%d (%s); using dense score",
                i, i + len(chunk) - 1, e,
            )
            s = [d["score"] for d in docs[i:i + batch_pairs]]
        scores.extend(float(x) for x in s)
    return scores

def main(
    model_name: str,
    dataset_path: str,
    faiss_index: str,
    split: str,
    out_path: str,
    faiss_meta: str,
    batch: int = 6,
    max_new: int = 768,
    seed: int = 42,
    search_k: int = 200,
    k_final: int = 5,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_split(dataset_path, split)
    if isinstance(ds[0], str):
        raise ValueError("Dataset rows are strings – supply JSONL records.")

    # ── W&B ───────────────────────────────────────────────────────────
    if wandb_project:
        if wandb is None:
            raise ImportError("install wandb or drop --wandb_project")
        wandb.init(project=wandb_project, name=wandb_run_name or f"A2_{Path(dataset_path).stem}_{split}")

    # ── Model & tokenizer ────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")

    multi_gpu = int(os.getenv("WORLD_SIZE", "1")) > 1
    print(f"Using {device} with multi-GPU={multi_gpu} for model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.eos_token_id

    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()

    # ── Retrieval Model ────────────────────────────────────────────
    retriever_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    retriever = BGERetriever(
        embedding_model=retriever_model,
        index_path=faiss_index,
        metadata_path=faiss_meta,
        device=device,
    )


    stop_list = StoppingCriteriaList([StopAfterAnswer(tok)])
    gen_cfg = dict(max_new_tokens=max_new, do_sample=False, stopping_criteria=stop_list)

    recs: List[dict] = []
    for start in track(range(0, len(ds), batch), description="Generating"):
        rows = ds.select(range(start, min(start + batch, len(ds))))

        def _q(r):
            return r.get("question", r.get("problem", r.get("prompt")))
        
        questions = [_q(r) for r in rows]
        docs_batch: list[list[dict]] = retriever.search_batch(questions, k=search_k)


        final_docs_batch: list[list[dict]] = []
        retrieved_contexts: list[str] = []
        for question, docs in zip(questions, docs_batch):
            scores = colbert_scores_safe(
                question,
                docs,
                retriever.model,          # BGEM3FlagModel instance
                batch_pairs=16            # pairs per sub‑batch
            )
            for d, s in zip(docs, scores):
                d["rerank_score"] = s
            topk = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:k_final]
            final_docs_batch.append(topk)

            ctx = ""
            for i, d in enumerate(topk, start=1):
                ctx += f"question {i}:\n{d.get('problem')}\n"
                ctx += f"{d.get('solution_chunk') or d.get('text')}\n"
            retrieved_contexts.append(ctx)
        
        model.config.pad_token_id = tok.eos_token_id

        prompts = [
            CHAT_TEMPLATE.format(system=SYSTEM_MSG, question=q, retrieved_context=ctx)
            for q, ctx in zip(questions, retrieved_contexts)
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)


        with torch.inference_mode():
            outs = model.generate(**inputs, pad_token_id=tok.eos_token_id, **gen_cfg)
        completions = tok.batch_decode(outs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        for row, txt in zip(rows, completions):
            recs.append({
                "id": row.get("idx", row.get("id")),
                "prediction": strip_answer(txt),
                "raw": txt,
                "answer": row.get("answer", row.get("solution")),

            })
        if wandb_project and wandb and len(recs) % (batch * 5) == 0:
            wandb.log({"generated": len(recs)})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(recs, indent=2))
    if wandb_project and wandb:
        art = wandb.Artifact(Path(out_path).stem, type="predictions")
        art.add_file(str(out_path))
        wandb.log_artifact(art)
        wandb.finish()

    print(f"Done → {out_path} (records={len(recs)})")


if __name__ == "__main__":
    fire.Fire(main)
