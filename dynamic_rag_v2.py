from __future__ import annotations
import os, re, json, logging

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from datasets import load_dataset
from rich.console import Console
from rich.progress import track
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from FlagEmbedding import BGEM3FlagModel
from BGERetriever_v2 import BGERetriever

# ---------------------------------------------------------------------------

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
console = Console()

# ---------------------------------------------------------------------------
# Tool definition (Llama 3.1 "tools" schema)
# ---------------------------------------------------------------------------

search_tool = {
    "name": "web_search",
    "description": "Search the local FAISS/BGE index for relevant math passages.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Concise natural‑language search query."
            },
            "k": {
                "type": "integer",
                "description": "Number of passages to retrieve.",
                "default": 100
            }
        },
        "required": ["query"]
    }
}
TOOLS = [search_tool]

# ---------------------------------------------------------------------------
# BGEM3 ColBERT reranker helpers
# ---------------------------------------------------------------------------

def _colbert_scores_safe(
    query: str,
    docs: list[dict],
    model: BGEM3FlagModel, # type: ignore
    batch_pairs: int = 16,
) -> list[float]:
    """Return ColBERT similarity scores for (query, doc) pairs.  
       Falls back to each doc's dense_score if GPU OOM."""
    pairs = [[query, d.get("solution_chunk") or d.get("text", "")] for d in docs]
    scores: list[float] = []
    for i in range(0, len(pairs), batch_pairs):
        try:
            chunk_scores = model.compute_score(
                pairs[i : i + batch_pairs],
                batch_size=batch_pairs,
            )["colbert"]
        except RuntimeError as e:
            logging.warning("ColBERT OOM on docs %d–%d → fallback to dense_score", i, i + batch_pairs - 1)
            chunk_scores = [d.get("dense_score", 0.0) for d in docs[i : i + batch_pairs]]
        scores.extend(float(s) for s in chunk_scores)
    return scores


def _rerank_colbert(query: str, docs: list[dict], model: BGEM3FlagModel, top_k: int = 5) -> list[dict]: # type: ignore
    """Return top_k docs after ColBERT reranking."""
    if not docs or model is None:
        return docs[:top_k]
    scores = _colbert_scores_safe(query, docs, model)
    for d, s in zip(docs, scores):
        d["rerank_score"] = s
    return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_k]

# ---------------------------------------------------------------------------
# Custom stopping criterion: halt as soon as we see </SEARCH>
# ---------------------------------------------------------------------------

class _StopOnString(StoppingCriteria):
    def __init__(self, tokenizer, stop_strs: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strs = stop_strs
        self.buf = ""

    def __call__(self, input_ids, scores, **kwargs):  # type: ignore
        # update running buffer with the last generated token
        self.buf += self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True)
        # keep buffer small
        if len(self.buf) > 200:
            self.buf = self.buf[-200:]
        for s in self.stop_strs:
            if s in self.buf:
                return True
        return False

# ---------------------------------------------------------------------------
# Stop on well‑formed JSON tool call (containing "tool_name")
# ---------------------------------------------------------------------------
class _StopOnToolJSON(StoppingCriteria):
    """
    Stop generation as soon as a well‑formed one‑level JSON tool call
    (containing the key "tool_name") is closed. This lets us pause mid‑
    generation and execute the tool before the model keeps reasoning.
    """
    def __init__(self, tokenizer, key: str = "\"tool_name\""):
        super().__init__()
        self.tokenizer = tokenizer
        self.key = key
        self.buf = ""

    def __call__(self, input_ids, scores, **kwargs):  # type: ignore
        # Append the latest token
        self.buf += self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True)
        # keep memory small
        if len(self.buf) > 1000:
            self.buf = self.buf[-1000:]

        # Only check if we have potentially started a JSON object that has "tool_name"
        if self.key in self.buf:
            # Find the last '{' and do a very light brace balance from there
            start = self.buf.rfind("{")
            if start != -1:
                snippet = self.buf[start:]
                bal = 0
                for ch in snippet:
                    if ch == "{":
                        bal += 1
                    elif ch == "}":
                        bal -= 1
                        if bal == 0:
                            # Closed a full JSON object
                            return True
        return False

# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------

def load_split(path: str, split: str):
    """Accept file, directory, recursive tree or HF dataset name."""
    import glob, os

    if os.path.isfile(path):
        return load_dataset("json", data_files={split: path}, split=split)

    if os.path.isdir(path):
        f = os.path.join(path, f"{split}.jsonl")
        if os.path.isfile(f):
            return load_dataset("json", data_files={split: f}, split=split)
        globbed = glob.glob(os.path.join(path, "**", f"{split}.jsonl"), recursive=True)
        if globbed:
            return load_dataset("json", data_files={split: globbed}, split=split)

    # HF hub dataset
    return load_dataset(path, split=split)

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

def get_initial_prompt(benchmark: str, problem: str) -> str:
    """
    Return an enhanced system prompt with clear, detailed instructions
    on how and when to call the `web_search` tool.
    """
    if benchmark == "gsm8k":
        sys = (
            "You are an expert math tutor who explains word‑problem solutions step‑by‑step.\n\n"
            "SOLVING PROCEDURE:\n"
            "1. **Understand** the question: identify what is asked and the given data.\n"
            "2. **Plan**: outline the computation in clear, numbered steps.\n"
            "3. **Search** if you need external facts (formulas, examples, definitions).\n"
            "   Use a JSON tool call exactly in this format:\n"
            '   {\"tool_name\":\"web_search\",\"arguments\":{\"query\":\"your specific query\"}}\n'
            "4. **Compute** each step, showing all intermediate work.\n"
            "5. **Verify** the result for sanity (units, reasonable magnitude).\n"
            "6. **Conclude** with:  **The final answer is: [number]**\n\n"
            "WHEN TO SEARCH:\n"
            "• If you can’t recall a formula or need a worked example.\n"
            "• Example good queries:  \"compound interest formula\", \"ratio word‑problem example\".\n"
            "• You may issue multiple searches; after each tool result, continue reasoning.\n\n"
            "ILLUSTRATIVE EXAMPLE (tool usage):\n"
            "User:  What is the LCM of 6 and 15?\n"
            'Assistant: {\"tool_name\":\"web_search\",\"arguments\":{\"query\":\"lcm of 6 and 15\"}}\n'
            "Tool(web_search): * 6 = 2·3  → …\n"
            "Assistant:  Using the context, the LCM is 30.\n"
            "The final answer is: 30"
        )
    else:  # math or other
        sys = (
            "You are an advanced mathematician solving competition‑level problems rigorously.\n\n"
            "APPROACH:\n"
            "1. **Analyze** the problem: determine the domain and target.\n"
            "2. **Recall** relevant theorems or techniques.\n"
            "3. **Search** for precise statements or example proofs if memory is uncertain.\n"
            "   Call the tool with JSON:\n"
            '   {\"tool_name\":\"web_search\",\"arguments\":{\"query\":\"your theorem or concept\"}}\n'
            "4. **Derive** the solution step‑by‑step, citing context where helpful.\n"
            "5. **Present** the final result clearly.\n"
            "6. Conclude with:  **The final answer is: [answer]**\n\n"
            "SEARCH STRATEGY:\n"
            "• Look up theorem statements, standard lemmas, or exemplar problems.\n"
            "• Example queries:  \"Eisenstein criterion example\", \"AM‑GM inequality proof\".\n"
            "• Multiple searches are allowed; integrate retrieved snippets into reasoning.\n\n"
            "ILLUSTRATIVE EXAMPLE (tool usage):\n"
            "User:  Prove that  \\sum a_i^2 ≥ (\\sum a_i)^2/n.\n"
            'Assistant: {\"tool_name\":\"web_search\",\"arguments\":{\"query\":\"cauchy schwarz inequality statement\"}}\n'
            "Tool(web_search): * C‑S: (Σa_i^2)(Σ1^2) ≥ (Σa_i)^2 …\n"
            "Assistant:  By C‑S, we have … Therefore the inequality holds.\n"
            "The final answer is: Proven."
        )

    return sys

# ---------------------------------------------------------------------------
# Helper: format retrieved docs into a short <ctx> block
# ---------------------------------------------------------------------------

def _format_passages(docs: list[dict], top_k: int = 3) -> str:
    lines = []
    for d in docs[:top_k]:
        q = d.get("question", d.get("problem", ""))[:150].replace("\n", " ")
        a = d.get("solution_chunk", d.get("solution", ""))[:220].replace("\n", " ")
        lines.append(f"* {q}\n  → {a}")
    return "\n".join(lines) if lines else "(no passages)"

# ---------------------------------------------------------------------------
# Sanitize messages so tokenizer.apply_chat_template won't choke on "tool"
# roles or None contents.
# ---------------------------------------------------------------------------
def _sanitize_for_template(messages: List[Dict]) -> List[Dict]:
    safe: List[Dict] = []
    for m in messages:
        role = m.get("role", "assistant")
        if role not in ("user", "assistant", "system"):
            # Treat unknown roles (e.g., "tool") specially
            if role == "tool":
                name = m.get("name", m.get("tool_name", "tool"))
                content = m.get("content")
                if content is None:
                    # Probably a tool call object – stringify it
                    payload = {k: v for k, v in m.items() if k != "role"}
                    safe.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
                else:
                    safe.append({"role": "user", "content": f"Tool[{name}] says:\n{content}"})
            else:
                # Fallback: stringify the whole dict
                safe.append({"role": "assistant", "content": json.dumps(m, ensure_ascii=False)})
        else:
            safe.append({"role": role, "content": m.get("content", "")})
    return safe

# ---------------------------------------------------------------------------
# Fallback chat: manual chat template application (no pipeline dependency)
# ---------------------------------------------------------------------------
def _simple_chat(
    model,
    tokenizer,
    messages,
    max_new_tokens: int = 512,
    stop_on_tool: bool = True,
):
    """
    Build a chat prompt via tokenizer.apply_chat_template and generate.
    If stop_on_tool=True, attach a stopping criterion that halts as soon
    as a JSON tool call closes. The model is expected to emit either:
      • normal assistant text, or
      • a JSON tool call object: {"tool_name": "...", "arguments": {...}}
    """
    # Make sure the template only sees roles it understands
    safe_messages = _sanitize_for_template(messages)
    prompt = tokenizer.apply_chat_template(
        safe_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    stopping_criteria = None
    if stop_on_tool:
        stopping_criteria = StoppingCriteriaList([_StopOnToolJSON(tokenizer)])

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    gen_text = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Try to parse as tool call (first well‑formed JSON object with "tool_name")
    try:
        candidate = json.loads(gen_text)
        if isinstance(candidate, dict) and "tool_name" in candidate:
            return {"role": "tool", **candidate}
    except Exception:
        pass

    return {"role": "assistant", "content": gen_text}

# ---------------------------------------------------------------------------
# STREAMING RAG ENGINE
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# STREAMING RAG ENGINE
# ---------------------------------------------------------------------------
def chat_with_tools(
    system_prompt: str,
    user_prompt: str,
    retriever,
    tokenizer,
    model,
    k_dense: int = 100,
    k_final: int = 3,
    debug: bool = False,
    max_turns: int = 4,
    tool_gen_tokens: int = 512,
    answer_gen_tokens: int = 1024,
    max_json_retries: int = 3,
    max_bad_query_retries: int = 3,
    min_query_chars: int = 15,
    min_query_words: int = 4,
    allow_multi_tools: bool = True,
    max_tool_calls: int = 2,
) -> Tuple[str, list[dict]]:
    """
    Tool‑aware chat loop. The model can emit a tool call:
      { "tool_name": "web_search", "arguments": { "query": "...", "k": 120 } }
    We execute, inject the passages, and continue.
    Returns (assistant_text, trace_of_calls)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    trace: list[dict] = []
    tool_calls = 0

    turns = 0
    bad_json_retry = 0
    bad_query_retry = 0

    def _bad_query(q: str) -> bool:
        q_strip = q.strip()
        if not q_strip:
            return True
        words = q_strip.split()
        alpha = sum(ch.isalpha() for ch in q_strip)
        # Basic length / signal checks
        if (
            len(q_strip) < max(min_query_chars, 18)
            or len(words) < max(min_query_words, 5)
            or (alpha / max(len(q_strip), 1)) < 0.7
        ):
            return True
        # Block obvious echoes of the original GSM-style prompt (these shouldn't need search)
        bad_starts = (
            "janet", "a robe", "josh", "james", "every day", "kylar",
            "toulouse", "carla", "john", "eliza"
        )
        low = q_strip.lower()
        if any(low.startswith(bs) for bs in bad_starts):
            return True
        # Looks fine
        return False

    while turns < max_turns:
        turns += 1

        # Always use our manual wrapper so stopping criteria are honored
        resp = _simple_chat(
            model,
            tokenizer,
            messages,
            max_new_tokens=tool_gen_tokens if tool_calls == 0 else answer_gen_tokens,
            stop_on_tool=(allow_multi_tools and tool_calls < max_tool_calls),
        )

        # ------------------------- TOOL BRANCH -------------------------
        if resp.get("role") == "tool":
            args = resp.get("arguments", {}) or {}
            query = str(args.get("query", "")).strip()
            k = int(args.get("k", k_dense))
            tool_calls += 1

            # Validate query; if junk, ask for a new one (bounded retries)
            if _bad_query(query):
                if bad_query_retry < max_bad_query_retries:
                    bad_query_retry += 1
                    messages.append({
                        "role": "system",
                        "content": (
                            f'The search query "{query}" is too short/noisy. '
                            "Output ONLY a single valid JSON tool call with a focused 8–20 word query that clearly states what you need."
                        )
                    })
                    continue
                else:
                    if debug:
                        console.print("[yellow]Giving up on malformed queries; continuing without search[/yellow]")
                    # Skip the search, push an assistant hint to keep going
                    messages.append({"role": "assistant", "content": "(Skipping search – continue reasoning.)"})
                    continue

            # Execute retrieval
            docs = retriever.search(query, k=k) if retriever else []
            top = _rerank_colbert(query, docs, getattr(retriever, "rerank_model", None), top_k=k_final)
            ctx = _format_passages(top, k_final)

            # Log & feed back to model
            trace.append({
                "query": query,
                "ctx": ctx,
                "num_docs_found": len(docs),
                "num_docs_used": len(top)
            })
            messages.append(resp)  # tool-call message itself
            messages.append({"role": "tool", "name": "web_search", "content": ctx})
            # Nudge the model to stop calling tools redundantly
            messages.append({
                "role": "system",
                "content": (
                    "Use the retrieved passages only as supporting evidence or examples. "
                    "Synthesize your own solution. Do NOT call the tool again unless you truly need new, different information. "
                    "Move toward the final answer now."
                )
            })
            if debug:
                console.print(f"[tool] query='{query}'  ctx_len={len(ctx)}", style="green")
            continue

        # --------------------- ASSISTANT TEXT BRANCH -------------------
        if resp.get("role") == "assistant":
            text = resp.get("content", "")

            # Try to parse inline JSON tool call inside assistant text
            if '{"tool_name"' in text:
                start_idx = 0
                parsed = False
                while True:
                    pos = text.find('{"tool_name"', start_idx)
                    if pos == -1:
                        break
                    stack = 0
                    end = None
                    for j in range(pos, len(text)):
                        if text[j] == '{':
                            stack += 1
                        elif text[j] == '}':
                            stack -= 1
                            if stack == 0:
                                end = j + 1
                                break
                    if end is None:
                        break
                    candidate = text[pos:end]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and obj.get("tool_name") == "web_search":
                            # Remove JSON snippet from assistant text
                            cleaned = (text[:pos] + text[end:]).strip()
                            if cleaned:
                                messages.append({"role": "assistant", "content": cleaned})
                            resp = {"role": "tool", **obj}
                            parsed = True
                            break
                    except Exception:
                        pass
                    start_idx = end if end else len(text)
                if parsed:
                    # Go back to loop top and handle as a normal tool call
                    continue

            # Assistant hinted at a tool call but JSON malformed -> ask for retry
            if '{"tool_name"' in text and bad_json_retry < max_json_retries:
                bad_json_retry += 1
                messages.append({
                    "role": "system",
                    "content": (
                        'Your previous message started a tool call but was not valid JSON. '
                        'Output ONLY a single JSON object of the form '
                        '{"tool_name":"web_search","arguments":{"query":"...","k":100}} and nothing else.'
                    )
                })
                continue

            # Normal assistant text: append & maybe finish
            messages.append(resp)
            if debug:
                console.print(f"[asst] {text[:120]}...", style="cyan")

            if "the final answer is" in text.lower():
                return text, trace

            # else keep looping
            continue

        # In case we somehow get an unexpected branch, just break
        if debug:
            console.print("[red]Unexpected response format; breaking[/red]")
        break

    # Reached max turns without an explicit final answer
    if debug:
        console.print(f"[yellow]Max turns ({max_turns}) reached without final answer[/yellow]")
    last_txt = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            last_txt = m["content"]
            break
    if last_txt and "final answer" not in last_txt.lower():
        # Ask the model to state the final answer explicitly
        messages.append({"role": "system", "content": "State the result succinctly: **The final answer is: [number/expression]**"})
        resp2 = _simple_chat(model, tokenizer, messages, max_new_tokens=128, stop_on_tool=False)
        if resp2.get("role") == "assistant":
            last_txt = resp2.get("content", last_txt)
    return last_txt, trace

# ---------------------------------------------------------------------------
# ANSWER EXTRACTION
# ---------------------------------------------------------------------------

def _latex_frac_to_str(s: str) -> Optional[str]:
    m = re.search(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None

def clean_answer(a: str) -> Optional[str]:
    """
    Normalize an extracted answer token:
    - Accept integers, decimals, simple fractions like 3/5
    - Accept LaTeX fractions \frac{a}{b}
    - Strip $ \boxed{} and surrounding whitespace/punctuation
    """
    if not a:
        return None
    # Unbox & trim
    a = a.strip()
    a = re.sub(r"^\$?\\boxed\{([^}]*)\}\$?$", r"\1", a)
    # Try LaTeX frac first
    frac = _latex_frac_to_str(a)
    if frac:
        return frac

    # Remove leading $ and other junk
    a = re.sub(r"^[^\d\-]+", "", a)
    # If there is a LaTeX frac inside
    frac = _latex_frac_to_str(a)
    if frac:
        return frac

    # Simple fraction or number
    if re.fullmatch(r"-?\d+/\d+", a):
        return a
    if re.fullmatch(r"-?\d+(?:\.\d+)?", a):
        return a
    return None

def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None

    # 1) Prefer explicit phrases
    explicit_patterns = [
        r"final answer is[:\s]*([^\.\n]+)",
        r"\\boxed\{([^}]+)\}",
        r"answer[:\s]*([^\.\n]+)",
    ]
    for p in explicit_patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            cand = clean_answer(m.group(1))
            if cand:
                return cand

    # 2) Look for LaTeX fractions anywhere
    m = re.search(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    # 3) Fallback: grab last numeric or fraction token (prefer fractions)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("* ")]
    stripped = "\n".join(lines)
    tokens = re.findall(r"-?\d+/\d+|-?\d+(?:\.\d+)?", stripped)
    if tokens:
        # if there is any fraction, prefer the last fraction
        fracs = [t for t in tokens if "/" in t]
        if fracs:
            return clean_answer(fracs[-1])
        return clean_answer(tokens[-1])
    return None

# ---------------------------------------------------------------------------
# SINGLE PROBLEM SOLVER
# ---------------------------------------------------------------------------

def solve(problem: str, retriever, model, tokenizer, bench: str, debug: bool = False, allow_heuristic_fallback: bool = False):
    system_prompt = get_initial_prompt(bench, problem)
    # First attempt: let the model decide
    resp, trace = chat_with_tools(
        system_prompt,
        problem,
        retriever,
        tokenizer,
        model,
        debug=debug,
        allow_multi_tools=True,
        max_tool_calls=2,
    )

    # Heuristic fallback removed to avoid noisy injections.

    return {
        "predicted_answer": extract_answer(resp),
        "full_reasoning": resp,
        "retrieval_triggered": bool(trace),
        "queries_generated": trace,
        "injections_made": trace,
        "total_steps": len(trace) + 1,
    }

# ---------------------------------------------------------------------------
# BENCHMARK RUNNER
# ---------------------------------------------------------------------------

def run_benchmark(
    benchmark="gsm8k",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path=None,
    num_samples=10,
    faiss_index="/local00/student/shakya/openmath_bge-m3_hnsw_index",
    faiss_meta="/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
    allow_heuristic_fallback: bool = False,
):
    # dataset discovery
    if dataset_path is None:
        root = f"./data/benchmarks/{benchmark}"
        hf   = "hendrycks/competition_math" if benchmark == "math" else "gsm8k"
        for p in [f"{root}/test.jsonl", root, hf]:
            if os.path.exists(p) or not p.startswith("."):
                dataset_path = p
                break

    ds = load_split(dataset_path, "test")
    ds = ds.select(range(min(num_samples, len(ds)))) if num_samples > 0 else ds

    # model
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # retriever
    retriever = None
    if faiss_index and faiss_meta and BGEM3FlagModel and BGERetriever:
        emb = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        retriever = BGERetriever(embedding_model=emb, index_path=faiss_index, metadata_path=faiss_meta, device="cuda")
        retriever.rerank_model = emb  # needed for ColBERT reranking

    results = []
    console.print(f"[bold blue]Solving {len(ds)} {benchmark.upper()} problems…[/bold blue]")
    for i, ex in enumerate(track(ds, description="Progress")):
        q_key = next(k for k in ex if k in {"question", "problem", "prompt", "input"})
        a_key = next((k for k in ex if k in {"answer", "solution", "output", "target"}), None)
        out = solve(ex[q_key], retriever, model, tok, benchmark, debug=i < 3, allow_heuristic_fallback=allow_heuristic_fallback)
        out.update({"id": i, "question": ex[q_key], "ground_truth": ex.get(a_key, "")})
        results.append(out)

    out_path = f"./results/{benchmark}_streamrag.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Saved → {out_path}[/green]")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import fire; fire.Fire(run_benchmark)