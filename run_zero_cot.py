from __future__ import annotations

import json, os, re, random
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import List, Optional

import fire, torch
from datasets import load_dataset
from rich.progress import track
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore

# ---------------------------------------------------------------------------
# Llama‑3.1 prompt
# ---------------------------------------------------------------------------
SYSTEM_MSG = """
    You are an expert mathematician.

    Think step-by-step.  
    Write every reasoning step inside `<think> … </think>` blocks.  
    When you are completely done, produce **exactly one**

        <answer> FINAL_NUMERIC_ANSWER </answer>

    Nothing after `</answer>`.
"""

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

# ---------------------------------------------------------------------------

def main(
    model_name: str,
    dataset_path: str,
    split: str,
    out_path: str,
    batch: int = 6,
    max_new: int = 768,
    seed: int = 42,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    random.seed(seed); torch.manual_seed(seed)
    ds = load_split(dataset_path, split)
    if isinstance(ds[0], str):
        raise ValueError("Dataset rows are strings – supply JSONL records.")

    # ── W&B ───────────────────────────────────────────────────────────
    if wandb_project:
        if wandb is None:
            raise ImportError("install wandb or drop --wandb_project")
        wandb.init(project=wandb_project, name=wandb_run_name or f"A1_{Path(dataset_path).stem}_{split}")

    # ── Model & tokenizer ────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    stop_list = StoppingCriteriaList([StopAfterAnswer(tok)])
    gen_cfg = dict(max_new_tokens=max_new, do_sample=False, temperature=0.0, stopping_criteria=stop_list)

    recs: List[dict] = []
    for start in track(range(0, len(ds), batch), description="Generating"):
        rows = ds.select(range(start, min(start + batch, len(ds))))

        def _q(r):
            return r.get("question", r.get("problem", r.get("prompt")))
        prompts = [CHAT_TEMPLATE.format(system=SYSTEM_MSG, question=_q(r)) for r in rows]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.inference_mode():
            outs = model.generate(**inputs, **gen_cfg)
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
