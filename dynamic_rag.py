from __future__ import annotations

from datetime import date
import os, re, json, glob, logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager

import torch
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from FlagEmbedding import BGEM3FlagModel
from BGERetriever_v2 import BGERetriever


import wandb



os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Silence tqdm bars and noisy retriever logs
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("FlagEmbedding").setLevel(logging.ERROR)

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

@contextmanager
def suppress_tqdm():
    prev = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = prev

# ---------------------------------------------------------------------------
# Llama 3 role templates (raw text, no apply_chat_template)
# ---------------------------------------------------------------------------
LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{user}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

def build_llama_prompt(system_txt: str, user_txt: str) -> str:
    return LLAMA3_TEMPLATE.format(system=system_txt.strip(), user=user_txt.strip())

def append_block(prompt: str, role: str, content: str) -> str:
    return prompt + f"<|start_header_id|>{role}<|end_header_id|>\n{content.strip()}\n<|eot_id|>\n"

# ---------------------------------------------------------------------------
# Custom stopping criteria: halt at the first well-formed <search>...</search>
# ---------------------------------------------------------------------------
class StopOnSearchTag(StoppingCriteria):
    """Stop generation once a complete <search>...</search> block is closed.

    We look for the literal strings "<search>" and "</search>" in a rolling buffer.
    """
    def __init__(self, tokenizer, open_tag: str = "<search>", close_tag: str = "</search>"):
        super().__init__()
        self.tok = tokenizer
        self.open_tag = open_tag
        self.close_tag = close_tag
        self.buf = ""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # append latest token (decoded) to buffer
        self.buf += self.tok.decode(input_ids[0, -1:], skip_special_tokens=True)
        # keep memory bounded
        if len(self.buf) > 4000:
            self.buf = self.buf[-4000:]
        # strict check: need an opening and a closing tag, in that order
        start = self.buf.rfind(self.open_tag)
        if start != -1:
            end = self.buf.find(self.close_tag, start + len(self.open_tag))
            if end != -1:
                return True
        return False

# ---------------------------------------------------------------------------
# Retrieval helpers (no truncation)
# ---------------------------------------------------------------------------
def colbert_scores_safe(query: str, docs: List[dict], model: BGEM3FlagModel, batch_pairs: int = 16) -> List[float]:
    pairs = [[query, d.get("solution_chunk") or d.get("text", "")] for d in docs]
    scores: List[float] = []
    for i in range(0, len(pairs), batch_pairs):
        try:
            with suppress_tqdm():
                chunk_scores = model.compute_score(pairs[i:i+batch_pairs], batch_size=batch_pairs)["colbert"]
        except RuntimeError:
            logging.warning("ColBERT OOM on docs %d–%d; fallback dense_score", i, i+batch_pairs-1)
            chunk_scores = [d.get("dense_score", 0.0) for d in docs[i:i+batch_pairs]]
        scores.extend(float(s) for s in chunk_scores)
    return scores

def rerank_colbert(query: str, docs: List[dict], model: Optional[BGEM3FlagModel], top_k: int) -> List[dict]:
    if not docs or model is None:
        return docs[:top_k]
    scores = colbert_scores_safe(query, docs, model)
    for d, s in zip(docs, scores):
        d["rerank_score"] = s
    return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_k]

def format_passages(docs: List[dict], top_k: int) -> str:
    lines = []
    for d in docs[:top_k]:
        q = d.get("question", d.get("problem", d.get("title", ""))).replace("\n", " ")
        a = d.get("solution_chunk", d.get("solution", d.get("text", ""))).replace("\n", " ")
        lines.append(f"* {q}\n  → {a}")
    return "\n".join(lines) if lines else "(no passages)"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def system_prompt_gsm() -> str:
    return (
        "You are an expert mathematician who explains mathematical word-problem solutions step-by-step.\n\n"
        "Step-by-step:\n"
        "1. **Understand** the question & data.\n"
        "2. **Plan** in numbered steps.\n"
        "3. **Search** ONLY if you need external info (formulas/examples/defs).\n"
        "   Emit exactly one block in this form (reason first, then the tag):\n"
        "   <search>Your focused query</search>\n"
        "   After writing the tag, STOP generating so I can run it.\n"
        "4. **Compute** the steps, show intermediate values.\n"
        "5. **Verify** sanity (units, magnitude).\n"
        "6. End with: **The final answer is: X**"
    )

def system_prompt_math() -> str:
    return (
        "You are an expert mathematician solving competition problems rigorously.\n\n"
        "Step-by-step:\n"
        "1. **Analyze** the problem.\n"
        "2. **Recall** relevant theorems/techniques.\n"
        "3. **Search** precise statements/examples if unsure. Emit ONLY one tag:\n"
        "   <search>The exact theorem/lemma you need</search>\n"
        "   Put reasoning first, then the tag, then STOP.\n"
        "4. **Derive** step-by-step.\n"
        "5. Conclude with: **The final answer is: ...**"
    )

def get_system_prompt(bench: str) -> str:
    return system_prompt_gsm() if bench == "gsm8k" else system_prompt_math()

# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------
def _latex_frac_to_str(s: str) -> Optional[str]:
    m = re.search(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None

def clean_answer(a: str) -> Optional[str]:
    if not a:
        return None
    a = a.strip()
    a = re.sub(r"^\$?\\boxed\{([^}]*)\}\$?$", r"\1", a)
    frac = _latex_frac_to_str(a)
    if frac:
        return frac
    a = re.sub(r"^[^\d\-]+", "", a)
    frac = _latex_frac_to_str(a)
    if frac:
        return frac
    if re.fullmatch(r"-?\d+/\d+", a):
        return a
    if re.fullmatch(r"-?\d+(?:\.\d+)?", a):
        return a
    return None

def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"final answer is[:\s]*([^\.\n]+)",
        r"\\boxed\{([^}]+)\}",
        r"answer[:\s]*([^\.\n]+)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            cand = clean_answer(m.group(1))
            if cand:
                return cand
    m = re.search(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    tokens = re.findall(r"-?\d+/\d+|-?\d+(?:\.\d+)?", text)
    if tokens:
        fracs = [t for t in tokens if "/" in t]
        return clean_answer(fracs[-1] if fracs else tokens[-1])
    return None

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def generate_with_stop(model, tokenizer, prompt: str, max_new_tokens: int, stop_for_search: bool) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
    stopping = None
    if stop_for_search:
        stopping = StoppingCriteriaList([StopOnSearchTag(tokenizer)])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()

SEARCH_OPEN, SEARCH_CLOSE = "<search>", "</search>"


def parse_search_block(text: str) -> Tuple[Optional[str], str]:
    """Return (query, cleaned_text_without_block). If no block, query=None."""
    if SEARCH_OPEN not in text:
        return None, text
    # find last opening tag to avoid partial earlier ones
    start = text.rfind(SEARCH_OPEN)
    end = text.find(SEARCH_CLOSE, start + len(SEARCH_OPEN))
    if start == -1 or end == -1:
        return None, text
    query = text[start + len(SEARCH_OPEN): end].strip()
    cleaned = (text[:start] + text[end + len(SEARCH_CLOSE):]).strip()
    return query, cleaned


def bad_query(q: str, min_chars=18, min_words=5) -> bool:
    qs = q.strip()
    if not qs:
        return True
    words = qs.split()
    alpha = sum(ch.isalpha() for ch in qs)
    if len(qs) < min_chars or len(words) < min_words or (alpha / max(len(qs), 1)) < 0.7:
        return True
    bad_starts = ("janet", "a robe", "josh", "james", "every day", "kylar", "toulouse", "carla", "john", "eliza")
    low = qs.lower()
    return any(low.startswith(bs) for bs in bad_starts)

# ---------------------------------------------------------------------------
# Core solve loop
# ---------------------------------------------------------------------------

def solve(problem: str,
          retriever,
          model,
          tokenizer,
          bench: str,
          k_dense: int,
          k_final: int,
          max_tool_calls: int = 2,
          tool_gen_tokens: int = 512,
          answer_gen_tokens: int = 2048,
          debug: bool = False) -> Dict:

    sys_prompt = get_system_prompt(bench)
    prompt = build_llama_prompt(sys_prompt, problem)

    tool_calls = 0
    trace: List[dict] = []
    final_text = ""
    turns = 0

    while turns < 8:
        turns += 1
        stop_for_search = tool_calls < max_tool_calls
        gen = generate_with_stop(
            model, tokenizer, prompt,
            max_new_tokens=tool_gen_tokens if stop_for_search else answer_gen_tokens,
            stop_for_search=stop_for_search,
        )

        query, cleaned_text = parse_search_block(gen)

        if query is not None:
            tool_calls += 1
            if bad_query(query):
                if debug:
                    console.print(f"[yellow]Bad query rejected: {query}[/yellow]")
                # keep whatever the model wrote before the tag (if any)
                if cleaned_text:
                    prompt = append_block(prompt, "assistant", cleaned_text)
                # instruct the model how to fix the query and show the previous one
                msg = (
                    "The query you produced was invalid (too short/noisy or it parrots the prompt).\n"
                    f'Previous query: "{query}"\n'
                    "Write exactly ONE improved <search>…</search> block and nothing else.\n"
                    "Constraints: 8–20 words, start with a noun phrase, no pronouns like I/you, "
                    "no copying numbers/text directly from the problem, and no equations or answers."
                )
                prompt = append_block(prompt, "system", msg)
                prompt = append_block(prompt, "assistant", "")
                continue

            docs = retriever.search(query, k=k_dense) if retriever else []
            top = rerank_colbert(query, docs, getattr(retriever, "rerank_model", None), k_final)
            ctx = format_passages(top, k_final)

            trace.append({
                "query": query,
                "ctx": ctx,
                "num_docs_found": len(docs),
                "num_docs_used": len(top),
            })

            # add model's reasoning (cleaned_text) then the tool block
            if cleaned_text:
                prompt = append_block(prompt, "assistant", cleaned_text)
            prompt = append_block(prompt, "tool", ctx)
            prompt = append_block(prompt, "system",
                "Use retrieved passages only as support/examples. Synthesize your own solution. "
                "Do NOT search again unless absolutely necessary. Move toward the final answer now.")
            prompt = append_block(prompt, "assistant", "")
            continue
        else:
            final_text = cleaned_text if cleaned_text else gen
            prompt = append_block(prompt, "assistant", final_text)
            if "final answer is" in final_text.lower():
                break
            prompt = append_block(prompt, "system", "State succinctly: **The final answer is: [value]**")
            prompt = append_block(prompt, "assistant", "")
            continue

    # append queries used to the reasoning text
    queries_used = [t["query"] for t in trace]
    full_reasoning_out = final_text
    if queries_used:
        full_reasoning_out += "\n\n---\nSearch queries used:\n" + "\n".join(f"- {q}" for q in queries_used)

    ans = extract_answer(final_text)
    return {
        "predicted_answer": ans,
        "full_reasoning": full_reasoning_out,
        "queries_used": queries_used,
        "retrieval_triggered": bool(trace),
        "queries_generated": trace,
        "injections_made": trace,
        "total_steps": len(trace) + 1,
    }

# ---------------------------------------------------------------------------
# Dataset utils
# ---------------------------------------------------------------------------

def load_split_any(path: str, split: str):
    if os.path.isfile(path):
        return load_dataset("json", data_files={split: path}, split=split)
    if os.path.isdir(path):
        f = os.path.join(path, f"{split}.jsonl")
        if os.path.isfile(f):
            return load_dataset("json", data_files={split: f}, split=split)
        globbed = glob.glob(os.path.join(path, "**", f"{split}.jsonl"), recursive=True)
        if globbed:
            return load_dataset("json", data_files={split: globbed}, split=split)
    return load_dataset(path, split=split)

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    benchmark: str = "gsm8k",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    dataset_path: Optional[str] = None,
    num_samples: Optional[int] = -1,
    faiss_index: Optional[str] = "/local00/student/shakya/openmath_bge-m3_hnsw_index",
    faiss_meta: Optional[str]  = "/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
    k_dense: int = 100,
    k_final: int = 10,
    allow_gpu8bit: bool = True,
    tool_gen_tokens: int = 512,
    answer_gen_tokens: int = 2048,
    wandb_project: Optional[str] = None,
    wandb_run: Optional[str] = None,
):

    # dataset discovery ------------------------------------------------------
    if dataset_path is None:
        root = f"./data/benchmarks/{benchmark}"
        hf   = "hendrycks/competition_math" if benchmark == "math" else "gsm8k"
        for p in [f"{root}/test.jsonl", root, hf]:
            if os.path.exists(p) or not p.startswith("."):
                dataset_path = p
                break
    ds = load_split_any(dataset_path, "test")
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    # model & tokenizer ------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(load_in_8bit=True) if allow_gpu8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant,
    )
    model.eval()

    # retriever --------------------------------------------------------------
    retriever = None
    if faiss_index and faiss_meta and BGEM3FlagModel and BGERetriever:
        emb = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        retriever = BGERetriever(embedding_model=emb, index_path=faiss_index, metadata_path=faiss_meta, device="cuda")
        retriever.rerank_model = emb
    else:
        console.print("[yellow]Retriever disabled (missing index/meta or libs).[/yellow]")

    # wandb ------------------------------------------------------------------
    if wandb and wandb_project:
        wandb.init(project=wandb_project, name=wandb_run, config=dict(
            benchmark=benchmark, model=model_name, k_dense=k_dense, k_final=k_final,
            num_samples=num_samples, tool_gen_tokens=tool_gen_tokens, answer_gen_tokens=answer_gen_tokens
        ))
    else:
        console.print("[grey62]W&B disabled[/grey62]")

    results = []
    console.print(f"[bold blue]Solving {len(ds)} {benchmark.upper()} problems…[/bold blue]")
    for i, ex in enumerate(track(ds, description="Progress")):
        q_key = next(k for k in ex if k in {"question", "problem", "prompt", "input"})
        a_key = next((k for k in ex if k in {"answer", "solution", "output", "target"}), None)
        out = solve(
            problem=ex[q_key],
            retriever=retriever,
            model=model,
            tokenizer=tok,
            bench=benchmark,
            k_dense=k_dense,
            k_final=k_final,
            tool_gen_tokens=tool_gen_tokens,
            answer_gen_tokens=answer_gen_tokens,
            debug=(i < 2),
        )
        out.update({
            "id": i,
            "question": ex[q_key],
            "ground_truth": ex.get(a_key, ""),
        })
        results.append(out)

        if wandb and wandb_project:
            wandb.log({
                "idx": i,
                "question": ex[q_key],
                "pred": out["predicted_answer"],
                "gt": ex.get(a_key, ""),
                "retrieval_triggered": out["retrieval_triggered"],
                "steps": out["total_steps"],
                "acc": int(clean_answer(ex.get(a_key, "")) == out["predicted_answer"]) if a_key else None,
            })

    out_path = f"./results/{benchmark}_streamrag_{date.today().strftime('%Y%m%d_%H%M')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Saved → {out_path}[/green]")

    if wandb and wandb_project:
        wandb.save(out_path)

# ---------------------------------------------------------------------------
# Fire CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import fire
    fire.Fire({
        "run_benchmark": run_benchmark,
    })
