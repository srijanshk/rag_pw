from __future__ import annotations

import logging
import json, os, re, random
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import List, Optional, Dict

import fire, torch
from tqdm import tqdm
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
RETRIEVE_TRIGGER  = "<|retrieve|>"
RETRIEVED_START   = "<|retrieved|>"
RETRIEVED_END     = "<|end_retrieved|>"

# SYSTEM_MSG = (
#     "You are an expert mathematician.  "
#     "You may **only** use external examples if you are **unsure** how to proceed.  "
#     "If you need worked-out examples, output exactly `<|retrieve|>` followed by a concise query "
#     "(≤ 10 tokens) such as 'quadratic inequality' or 'compound interest'.  "
#     "You will then receive **one or two short solved examples**.  "
#     "After reading them, continue your own solution.  "
#     "If you already know the method, solve immediately **without** emitting `<|retrieve|>`.  "
#     "Your final answer must appear on a single line: `<answer>NUMERIC_ANSWER</answer>`."
# )
SYSTEM_MSG = (
    "You are a methodical mathematician. Your task is to solve the user's question by thinking step-by-step. "
    "Crucially, before attempting a solution, you MUST decide if you need to retrieve a relevant formula or a worked example. "
    "If you need external information, you MUST generate a search query by typing `<|retrieve|> short_query_here` on a new line. "
    "**DO NOT write comments like '# I need to search for...' or '# Search query:'. YOU MUST use the `<|retrieve|>` token to trigger the search. This is the only way to get information.**\n\n"
    "After you provide the query, you will receive examples between <|retrieved|> and <|end_retrieved|> tags. "
    "Use these examples to inform your step-by-step reasoning. "
    "If and only if you are absolutely certain you know the exact formula and method required, you may solve the problem directly without using `<|retrieve|>`. "
    "Always conclude your final answer on a new line in the format: `<answer>NUMERIC_ANSWER</answer>`.\n\n"
    "Here is an example of the correct way to work:\n"
    "Q: A ladder of length 5 meters is leaning against a wall. The base of the ladder is 3 meters away from the wall. How high up the wall does the ladder reach?\n"
    "A:\n"
    "Step 1: This problem involves a right-angled triangle. I need the formula relating the sides.\n"
    "<|retrieve|> Pythagorean theorem for right-angled triangle\n"
    "<|retrieved|>\n"
    "Ex: In a right-angled triangle, the square of the hypotenuse (c²) is equal to the sum of the squares of the other two sides (a² + b²).\n"
    "<|end_retrieved|>\n"
    "Step 2: The formula is a² + b² = c². The ladder is the hypotenuse (c = 5), and the base is one side (a = 3).\n"
    "Step 3: 3² + b² = 5² → 9 + b² = 25 → b² = 16.\n"
    "Step 4: b = √16 = 4.\n"
    "<answer>4</answer>\n\n"
    "Now, solve the following question."
)

CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n{system}\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n{question}\n<|eot_id|>"
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
specials = [RETRIEVE_TRIGGER, RETRIEVED_START, RETRIEVED_END]


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


class RetrieveCriteria(StoppingCriteria):
    """Stop when <|retrieve|> token is generated."""
    def __init__(self, tokenizer, trigger_id):
        super().__init__()
        self.tokenizer = tokenizer
        self.trigger_id = trigger_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.trigger_id

# ---------------------------------------------------------------------------
# Dataset loader (handles GSM8K & MATH JSONL)
# ---------------------------------------------------------------------------

def load_split(path: str, split: str):
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
    pairs = [[query, d.get("solution_chunk") or d.get("text", "")] for d in docs]
    scores = []
    for i in range(0, len(pairs), batch_pairs):
        try:
            sc = model.compute_score(
                pairs[i:i+batch_pairs],
                batch_size=batch_pairs,
                max_query_length=128
            )["colbert"]
        except RuntimeError as e:
            logging.warning("ColBERT OOM on %d–%d: %s → use dense",
                            i, i+batch_pairs-1, e)
            sc = [d["dense_score"] for d in docs[i:i+batch_pairs]]
        scores.extend(float(x) for x in sc)
    return scores


def main(
    model_name: str,
    dataset_path: str,
    faiss_index: str,
    split: str,
    out_path: str,
    faiss_meta: str,
    batch: int = 6,
    max_new: int = 1024,
    seed: int = 42,
    search_k: int = 200,
    k_final: int = 5,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    max_retries: int = 3,
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
    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.eos_token_id

    model.config.pad_token_id = tok.pad_token_id
    # model.generation_config.temperature = None
    # model.generation_config.top_p = None
    model.eval()

    new_tokens = [t for t in specials if t not in tok.additional_special_tokens]
    if new_tokens:
        tok.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tok),  pad_to_multiple_of=8,  mean_resizing=False)

    retrieve_id = tok.convert_tokens_to_ids(RETRIEVE_TRIGGER)

    # ── Retrieval Model ────────────────────────────────────────────
    retriever_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    retriever = BGERetriever(
        embedding_model=retriever_model,
        index_path=faiss_index,
        metadata_path=faiss_meta,
        device=device,
    )
    
    def rerank(query: str, docs: List[Dict], k: int = 2) -> List[Dict]:
        """ColBERT re-rank using your BGEM3FlagModel."""
        scores = colbert_scores_safe(query, docs, retriever.model, batch_pairs=16)
        for d, s in zip(docs, scores):
            d["rerank_score"] = s
        return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:k]

    stop_list = StoppingCriteriaList([
        RetrieveCriteria(tok, retrieve_id),
        StopAfterAnswer(tok),
    ])
    gen_cfg = dict(max_new_tokens=max_new, do_sample=True, temperature=0.2, top_p=0.95, stopping_criteria=stop_list)

    recs: List[dict] = []
    for row in tqdm(ds, desc="Generating", total=len(ds)):
        question = row.get("question", row.get("problem"))
        prompt = CHAT_TEMPLATE.format(system=SYSTEM_MSG, question=question)

        # We build the prompt turn-by-turn in this loop
        prompt_so_far = prompt
        completion = ""
        retrieval_history = []
        
        # The stop list should only contain the answer-stopper now
        stop_list = StoppingCriteriaList([StopAfterAnswer(tok)])

        for retry in range(max_retries + 1):
            inputs = tok(
                prompt_so_far,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,  # Increased max_length to accommodate context
            ).to(device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                stopping_criteria=stop_list,
                pad_token_id=tok.eos_token_id,
            )

            # Decode only the newly generated text
            generated_text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)

            # Check if the model wants to retrieve
            if RETRIEVE_TRIGGER in generated_text and retry < max_retries:
                # Split the generation at the first retrieval trigger
                parts = generated_text.split(RETRIEVE_TRIGGER, 1)
                text_before_retrieve = parts[0]
                query_part = parts[1].split('\n')[0].strip()

                # If the query is empty, we stop and treat it as a final generation
                if not query_part:
                    completion = generated_text
                    break

                print(f"🔄 Retrieval triggered with query: '{query_part}'")
                
                # Perform retrieval and reranking
                docs = retriever.search(query_part, k=search_k)
                top_docs = rerank(query_part, docs, k=k_final)
                ctx = "\n".join(
                    f"Ex: {d['problem'].splitlines()[0]} → {d.get('answer', '')}"
                    for d in top_docs
                )
                
                # Log the retrieval event
                retrieval_history.append({"query": query_part, "context": ctx})

                # Reconstruct the prompt for the next turn
                retrieval_block = (
                    f"{RETRIEVE_TRIGGER} {query_part}\n"
                    f"{RETRIEVED_START}\n{ctx}\n{RETRIEVED_END}\n"
                )
                # Append the text leading to the retrieval, plus the new context block
                prompt_so_far += text_before_retrieve + retrieval_block
                
                # Continue the loop to generate the final solution with the new context
                continue
            else:
                # No retrieval trigger OR max retries reached: this is the final answer
                # We must use skip_special_tokens=True for the final output processing
                completion = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                break

        # After the loop, save the results
        recs.append({
            "id": row.get("idx", row.get("id")),
            "prediction": strip_answer(completion),
            "raw": completion,
            "answer": row.get("answer", row.get("solution")),
            "retrieved_queries": [h['query'] for h in retrieval_history],
            "retrieved_examples": [h['context'] for h in retrieval_history],
        })

        if wandb_project and len(recs) % 10 == 0:
            wandb.log({"generated": len(recs)})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(recs, indent=2))
    if wandb_project:
        art = wandb.Artifact(Path(out_path).stem, type="predictions")
        art.add_file(str(out_path))
        wandb.log_artifact(art)
        wandb.finish()
    print(f"Done → {out_path} ({len(recs)} samples)")

if __name__ == "__main__":
    fire.Fire(main)
