from __future__ import annotations

from datetime import date, datetime
import os
import re
import json
import glob
import logging
import random
import csv
import statistics as st
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


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3, 4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
        # stop after the FIRST closed </search>
        if self.open_tag in self.buf and self.close_tag in self.buf:
            start = self.buf.find(self.open_tag)
            end = self.buf.find(self.close_tag, start + len(self.open_tag))
            if start != -1 and end != -1:
                return True
        return False

# ---------------------------------------------------------------------------
# Retrieval helpers (no truncation)
# ---------------------------------------------------------------------------
def colbert_scores_safe(query: str, docs: List[dict], model: BGEM3FlagModel, batch_pairs: int = 16) -> List[float]:
    pairs = [[query, d.get("solution_chunk") or d.get("text", "")]
             for d in docs]
    scores: List[float] = []
    for i in range(0, len(pairs), batch_pairs):
        try:
            with suppress_tqdm():
                chunk_scores = model.compute_score(
                    pairs[i:i+batch_pairs], batch_size=batch_pairs)["colbert"]
        except RuntimeError:
            logging.warning(
                "ColBERT OOM on docs %d–%d; fallback dense_score", i, i+batch_pairs-1)
            chunk_scores = [d.get("dense_score", 0.0)
                            for d in docs[i:i+batch_pairs]]
        scores.extend(float(s) for s in chunk_scores)
    return scores

def rerank_colbert(query: str, docs: List[dict], model: Optional[BGEM3FlagModel], top_k: int) -> List[dict]:
    if not docs or model is None:
        return docs[:top_k]
    scores = colbert_scores_safe(query, docs, model)
    for d, s in zip(docs, scores):
        d["rerank_score"] = s
    return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_k]

def format_passages(docs: List[dict]) -> str:
    lines = []
    for d in docs:
        q = d.get("full_problem", d.get(
            "problem", d.get("title", ""))).replace("\n", " ")
        a = d.get("full_solution", d.get(
            "solution", d.get("text", ""))).replace("\n", " ")
        lines.append(f"* Q: {q}\n  → A: {a}")
    return "\n".join(lines) if lines else "(no passages)"


# ---------------------------------------------------------------------------
# Raw context masking & injection helpers
# ---------------------------------------------------------------------------
def _mask_text_for_injection(s: str) -> str:
    """Mask numbers and common answer markers to reduce leakage when injecting raw context."""
    if not s:
        return s
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "#", s)
    s = s.replace("\\boxed", "").replace("The final answer is", "")
    return s


# ---------------------------------------------------------------------------
# Format masked raw context for injection (top-3 docs)
# ---------------------------------------------------------------------------
def format_raw_context_for_injection(docs: List[dict], max_chars_per_doc: int = 600) -> str:
    """Format raw retrieved docs for injection, with masking and truncation."""
    blocks = []
    for i, d in enumerate(docs[:3]):  # cap to top-3 raw docs
        title = (d.get("full_problem") or d.get("problem") or d.get("title") or "").strip().replace("\n", " ")
        txt = get_doc_text(d).strip()
        if len(txt) > max_chars_per_doc:
            txt = txt[:max_chars_per_doc] + "…"
        txt = _mask_text_for_injection(txt)
        header = f"[Doc {i+1}] {title}" if title else f"[Doc {i+1}]"
        blocks.append(f"{header}\n{txt}")
    return "\n\n".join(blocks) if blocks else "(no raw passages)"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def system_prompt_gsm() -> str:
    return (
        "You are an expert mathematician that solves problems step-by-step while minimizing tool use.\n"
        "Follow this protocol exactly; violations are errors.\n\n"
        "== PROBLEM DECOMPOSITION (First Turn Only) ==\n"
        "Before starting the action loop, your first output must be a structured breakdown of the problem:\n"
        "1. **Variables:** List every number given in the problem and what it represents.\n"
        "2. **Goal:** State precisely what the question is asking you to find, including the units.\n"
        "3. **Plan:** Write a numbered, step-by-step plan of the calculations needed to get from the variables to the goal.\n\n"
        "== ACTION LOOP (Subsequent Turns) ==\n"
        "After the decomposition, at each step k, do three things:\n"
        "1) REASON (short): Execute the next step of your plan. Show your work.\n"
        "2) SANITY CHECK: Before deciding, quickly check if your result is logical. If it seems nonsensical (e.g., negative distance, a part is larger than the whole), state the error and announce that you will revise your plan or re-read the problem in the next step.\n"
        "3) DECIDE (gap check): Search ONLY if you need a precise statement/definition/identity, a named theorem/lemma, or a standard method outline you cannot confidently reconstruct. Do NOT search for routine arithmetic/algebra, unit conversions, floor/ceil counting, or facts you can derive.\n"
        "4) ACT: If no gap and the sanity check passes, continue. If there is a gap or a sanity check failure, emit a control block and STOP this turn.\n\n"
        "== CONTROL GRAMMAR (machine-parsable) ==\n"
        "Emit exactly one block per step:\n"
        "<control>{\"step\":k,\"action\":\"continue\"|\"search\"|\"finish\",\"need_type\":null|\"definition\"|\"identity\"|\"named_theorem\"|\"method_outline\",\"can_derive\":true|false,\"confidence_0_1\":0.0,\"why\":\"minimal reason\",\"query\":null|\"used only when action='search'\",\"query_alternatives\":[\"opt1\",\"opt2\"],\"chosen_idx\":0,\"asked_quantity\":\"what is being asked (type/units)\"}</control>\n"
        "If action=\"search\", immediately follow with EXACTLY ONE:\n<search>Descriptive canonical query; name concept + goal; no equations or instance numbers.</search>\nThen STOP.\n\n"
        "== QUERY STYLE ==\n"
        "Name the canonical concept + goal (e.g., 'Chinese Remainder Theorem — statement and simple application'). Avoid copying instance numbers; never include equations or answers.\n\n"
        "== EVIDENCE INTEGRATION (next turn) ==\n"
        "Verify snippets match your need; list only useful statements; continue derivation succinctly.\n\n"
        "== FINISHING ==\n"
        "When you have the numeric result for the asked quantity, emit: **The final answer is: X**\n\n"
        "== BENCH_POLICY: GSM8K ==\n"
        "Treat most content as derivable; search rarely and only for named statements you cannot reconstruct.\n"
    )


def system_prompt_math() -> str:
    return (
        "You are an expert mathematician solving competition problems rigorously, with minimal tool use.\n"
        "Follow this protocol exactly; violations are errors.\n\n"
        "== ACTION LOOP ==\n"
        "At each step k, do three things:\n"
        "1) REASON (short).\n"
        "2) DECIDE (gap check): Search ONLY for a precise theorem/lemma/identity/definition or standard method outline you cannot confidently reconstruct.\n"
        "3) ACT: No gap → continue. Gap → emit <control>{…}</control>, then one <search>…</search> and STOP this turn.\n\n"
        "== CONTROL GRAMMAR (machine-parsable) ==\n"
        "<control>{\"step\":k,\"action\":\"continue\"|\"search\"|\"finish\",\"need_type\":null|\"definition\"|\"identity\"|\"named_theorem\"|\"method_outline\",\"can_derive\":true|false,\"confidence_0_1\":0.0,\"why\":\"minimal reason\",\"query\":null|\"used only when action='search'\",\"query_alternatives\":[\"opt1\",\"opt2\"],\"chosen_idx\":0,\"asked_quantity\":\"what is being asked (type/units)\"}</control>\n"
        "If action=\"search\", immediately follow with EXACTLY ONE:\n<search>Descriptive canonical query; name concept + goal; no equations or instance numbers.</search>\nThen STOP.\n\n"
        "== QUERY STYLE ==\n"
        "Use canonical names (e.g., 'Vieta\\'s formulas — relation between roots and coefficients', 'Angle bisector theorem — statement and application to inradius', 'Inclusion–exclusion — standard formulation'). No equations or answers.\n\n"
        "== INTEGRATION & FINISH ==\n"
        "Integrate only the necessary statements; proceed to a correct solution.\n\n"
        "== RIGOR & VERIFICATION ==\n"
        "After a complex calculation (e.g., back-substitution in EEA, applying a multi-part formula), briefly pause to double-check the arithmetic. Before finishing, quickly review all steps to ensure they logically cohere and satisfy all problem constraints.\n\n"
        "When you have the final numeric answer, finish with **The final answer is: X**\n\n"
        "== BENCH_POLICY: MATH ==\n"
        "Search is allowed for exact statements and method outlines when you cannot reconstruct them confidently; otherwise derive.\n"
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
# Final-answer extraction & normalization (robust)
# ---------------------------------------------------------------------------
FINAL_ANSWER_PATTERNS = [
    re.compile(
        r'\*\*?\s*The\s+final\s+answer\s+is\s*:\s*(.+?)\s*\*\*', re.IGNORECASE),
    re.compile(
        r'(?:^|\n)\s*The\s+final\s+answer\s+is\s*:\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(
        r'(?:^|\n)\s*final\s*answer\s*[:\-–]\s*(.+?)(?:\n|$)', re.IGNORECASE),
]

def extract_final_answer_text(reasoning: str) -> Optional[str]:
    """
    Find the LAST occurrence of a 'final answer' declaration and return the raw text after the colon.
    Handles bold markdown and plain text. Returns None if not found.
    """
    last = None
    for pat in FINAL_ANSWER_PATTERNS:
        for m in pat.finditer(reasoning or ""):
            cand = (m.group(1) or "").strip()
            # only keep the current line and strip trailing fmt marks
            cand = cand.splitlines()[0].strip()
            cand = re.sub(r'[\s`*]+$', '', cand)
            if cand:
                last = cand
    return last


def normalize_final_answer(ans_text: str) -> str:
    """
    Normalize a raw final-answer text to a comparison-friendly string.
    Prefer robust cleaning (handles \boxed and \frac) before regex fallback.
    """
    ca = clean_answer(ans_text or "")
    if ca:
        return ca
    s = (ans_text or "").strip().replace(',', '').replace('$', '').replace('%', '')
    m = re.search(r'([-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)', s)
    return m.group(1) if m else s

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def generate_with_stop(model, tokenizer, prompt: str, max_new_tokens: int, stop_for_search: bool) -> str:
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=False).to(model.device)
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
    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


SEARCH_OPEN, SEARCH_CLOSE = "<search>", "</search>"
CONTROL_OPEN, CONTROL_CLOSE = "<control>", "</control>"
REASON_OPEN, REASON_CLOSE = "<search_reason>", "</search_reason>"
def parse_control_block(text: str) -> Tuple[Optional[str], str]:
    """Return (control_text, cleaned_text_without_control). If no block, control=None."""
    if CONTROL_OPEN not in text:
        return None, text
    start = text.rfind(CONTROL_OPEN)
    end = text.find(CONTROL_CLOSE, start + len(CONTROL_OPEN))
    if start == -1 or end == -1:
        return None, text
    control = text[start + len(CONTROL_OPEN): end].strip()
    cleaned = (text[:start] + text[end + len(CONTROL_CLOSE):]).strip()
    return control, cleaned

def parse_reason_block(text: str) -> Tuple[Optional[str], str]:
    """Return (reason_text, cleaned_text_without_reason_block). If no block, reason=None."""
    if REASON_OPEN not in text:
        return None, text
    start = text.rfind(REASON_OPEN)
    end = text.find(REASON_CLOSE, start + len(REASON_OPEN))
    if start == -1 or end == -1:
        return None, text
    reason = text[start + len(REASON_OPEN): end].strip()
    cleaned = (text[:start] + text[end + len(REASON_CLOSE):]).strip()
    return reason, cleaned


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

def normalize_control_json(obj: Optional[dict]) -> Optional[dict]:
    if not isinstance(obj, dict):
        return obj
    # Normalize common typos
    if "confidence_1.0" in obj and "confidence_0_1" not in obj:
        obj["confidence_0_1"] = obj.pop("confidence_1.0")
    return obj

# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------
def get_doc_text(d: dict) -> str:
    return d.get("solution_chunk") or d.get("solution") or d.get("text", "")


def cross_encoder_rescore(query: str, docs: List[dict], cross_encoder, final_k: int, batch_size: int = 16) -> List[dict]:
    """Rescore provided docs with a cross-encoder and return top final_k docs."""
    if not cross_encoder or not docs:
        return docs[:final_k]
    pairs = [[query, get_doc_text(d)] for d in docs]
    scores: List[float] = []
    for i in range(0, len(pairs), batch_size):
        with suppress_tqdm():
            try:
                # FlagReranker returns a list of floats
                sc = cross_encoder.compute_score(
                    pairs[i:i+batch_size], batch_size=batch_size)
            except Exception:
                sc = [0.0] * len(pairs[i:i+batch_size])
        scores.extend(float(s) for s in sc)
    for d, s in zip(docs, scores):
        d["ce_score"] = s
    return sorted(docs, key=lambda d: d.get("ce_score", 0.0), reverse=True)[:final_k]


# ---------------------------------------------------------------------------
# Retrieval distillation (summary + how to proceed)
# ---------------------------------------------------------------------------

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
          ) -> Dict:

    sys_prompt = get_system_prompt(bench)
    prompt = build_llama_prompt(sys_prompt, problem)

    tool_calls = 0
    trace: List[dict] = []
    final_text = ""
    turns = 0
    transcript: List[str] = []
    retrieval_attempts = 0
    retrieval_executed = 0

    while turns < max_tool_calls + 1:
        turns += 1
        stop_for_search = tool_calls < max_tool_calls
        gen = generate_with_stop(
            model, tokenizer, prompt,
            max_new_tokens=tool_gen_tokens if stop_for_search else answer_gen_tokens,
            stop_for_search=stop_for_search,
        )
        transcript.append(gen)

        # Extract control/reason first (if any), then the search tag; keep both for logging but strip them from continuation
        control_text, text_wo_control = parse_control_block(gen)
        reason_text, text_wo_reason = parse_reason_block(text_wo_control)
        query, cleaned_text = parse_search_block(text_wo_reason)

        if query is not None:
            tool_calls += 1
            retrieval_attempts += 1

            # Parse control JSON if present; otherwise fall back to <search_reason>
            reason_info = None
            if control_text:
                try:
                    reason_info = json.loads(control_text)
                except Exception:
                    reason_info = {"raw_control": control_text}
            elif reason_text:
                try:
                    reason_info = json.loads(reason_text)
                except Exception:
                    reason_info = {"raw": reason_text}

            reason_info = normalize_control_json(reason_info) if reason_info else None

            skip_due_to_derivable = False
            if bench == "gsm8k" and isinstance(reason_info, dict) and reason_info.get("can_derive") is True:
                skip_due_to_derivable = True

            original_query = query
            search_query: str = original_query

            # Run retrieval (unless derivable on GSM)
            docs = []
            if retriever and not skip_due_to_derivable:
                docs = retriever.search_and_rerank(
                    search_query,
                    top_k=k_dense,
                    top_k_final=k_final,
                )
            ctx_raw_inject = format_passages(docs)

            # Telemetry from retriever if available
            rinfo = getattr(retriever, "get_last_search_info", lambda: {})() if retriever else {}

            # Trace (log raw context + telemetry)
            trace.append({
                "query": search_query,
                "search_reason": reason_info,
                "ctx_raw": ctx_raw_inject,
                "num_docs_found": len(docs),
                "retrieval_info": rinfo,
                "skipped_due_to_derivable": skip_due_to_derivable,
            })

            # Provide only raw evidence to the model
            if not skip_due_to_derivable:
                prompt = append_block(prompt, "user", f"Retrieved evidence (raw, masked numbers):\n{ctx_raw_inject}")
                # If retrieval looked weak, gently nudge a rewrite on next turn
                if rinfo.get("low_confidence"):
                    prompt = append_block(
                        prompt,
                        "system",
                        "Retrieval appears weak. If needed, rewrite the query with canonical concept names and emit another <search>. Otherwise continue solving succinctly."
                    )

            # Also append any assistant continuation that followed the search tag
            if cleaned_text:
                prompt = append_block(prompt, "assistant", cleaned_text)

            retrieval_executed += 1
        else:
            final_text = cleaned_text if cleaned_text else gen
            prompt = append_block(prompt, "assistant", final_text)
            if "final answer is" in final_text.lower():
                # One-shot verification: ensure target/units match the question; fix if needed
                verify_prompt = append_block(
                    prompt, "system",
                    "Before finalizing: (1) Restate the exact quantity asked. (2) Check units/type (integer vs fraction vs real; sign; range). (3) Do a 1-line dimensional/invariants check (e.g., time=distance/rate; position on a line with signs for directions). (4) Ensure no step used instance numbers copied from retrieved text. If anything mismatches, correct it briefly. Then output ONLY: **The final answer is: X**"
                )
                verify_out = generate_with_stop(
                    model, tokenizer, verify_prompt,
                    max_new_tokens=min(256, answer_gen_tokens // 2),
                    stop_for_search=False,
                )
                prompt = append_block(prompt, "assistant", verify_out)
                transcript.append(verify_out)
                final_text = verify_out
                break
            prompt = append_block(
                prompt, "system", "State succinctly: **The final answer is: [value]**")
            prompt = append_block(prompt, "assistant", "")
            continue

    # append queries used to the reasoning text
    queries_used = [t["query"] for t in trace]
    # Stitch full reasoning from all turns (assistant generations and retrieved contexts)
    full_reasoning_out = "\n\n---\n".join(transcript)
    if queries_used:
        full_reasoning_out += "\n\n---\nSearch queries used:\n" + \
            "\n".join(f"- {q}" for q in queries_used)

    # Robust final-answer extraction from the full transcript (with fallbacks)
    pred_text = extract_final_answer_text(full_reasoning_out)
    predicted_answer = normalize_final_answer(pred_text) if pred_text else None
    pred_source = "final_tag" if pred_text else None

    # Fallback: try last assistant turn only
    if predicted_answer is None:
        legacy_pred_text = extract_final_answer_text(final_text or "")
        if legacy_pred_text:
            predicted_answer = normalize_final_answer(legacy_pred_text)
            pred_text = legacy_pred_text
        pred_source = pred_source or "final_tag_last_turn"

    # Last-resort guard: take the final numeric token in the overall reasoning if needed
    if predicted_answer is None:
        nums = re.findall(
            r'([-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)', full_reasoning_out or "")
        predicted_answer = nums[-1] if nums else None
    if predicted_answer is not None and pred_source is None:
        pred_source = "fallback_numeric"

    return {
        "predicted_answer": predicted_answer,
        "predicted_answer_raw": pred_text,
        "predicted_answer_source": pred_source,
        "full_reasoning": full_reasoning_out,
        "queries_used": queries_used,
        "retrieval_attempted": retrieval_attempts > 0,
        "retrieval_executed": retrieval_executed > 0,
        "retrieval_triggered": retrieval_executed > 0,
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
        globbed = glob.glob(os.path.join(
            path, "**", f"{split}.jsonl"), recursive=True)
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
    faiss_meta: Optional[str] = "/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
    k_dense: int = 100,
    k_final: int = 5,
    allow_gpu8bit: bool = True,
    tool_gen_tokens: int = 512,
    answer_gen_tokens: int = 2048,
    max_tool_calls: int = 2,
    wandb_project: Optional[str] = None,
    wandb_run: Optional[str] = None,
    out_path: Optional[str] = None,
):

    # dataset discovery ------------------------------------------------------
    if dataset_path is None:
        root = f"./data/benchmarks/{benchmark}"
        hf = "hendrycks/competition_math" if benchmark == "math" else "gsm8k"
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
        emb_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        retriever = BGERetriever(
            embedding_model=emb_model,
            index_path=faiss_index,                    
            metadata_path=faiss_meta,            
        )
    else:
        console.print(
            "[yellow]Retriever disabled (missing index/meta or libs).[/yellow]")

    # wandb ------------------------------------------------------------------
    if wandb and wandb_project:
        wandb.init(project=wandb_project, name=wandb_run, config=dict(
            benchmark=benchmark, model=model_name, k_dense=k_dense, k_final=k_final,
            num_samples=num_samples, tool_gen_tokens=tool_gen_tokens, answer_gen_tokens=answer_gen_tokens,
            max_tool_calls=max_tool_calls
        ))
    else:
        console.print("[grey62]W&B disabled[/grey62]")

    # Benchmark-specific runtime defaults
    if benchmark == "math":
        # Limit to a single tool turn to avoid duplicate searches in one problem
        if max_tool_calls > 1:
            max_tool_calls = 1

    console.print(
        f"[bold blue]Solving {len(ds)} {benchmark.upper()} problems…[/bold blue]")
    results = []
    for i, ex in enumerate(track(ds, description="Progress")):
        q_key = next(k for k in ex if k in {
                     "question", "problem", "prompt", "input"})
        a_key = next(
            (k for k in ex if k in {"answer", "solution", "output", "target"}), None)
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
            max_tool_calls=max_tool_calls,
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

    out_path = out_path if out_path else f"./results/{benchmark}_streamrag_{date.today().strftime('%Y%m%d_%H%M')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Saved → {out_path}[/green]")

    if wandb and wandb_project:
        wandb.save(out_path)


# ---------------------------------------------------------------------------
# Results analyzer (EM, retrieval stats, simple breakdowns)
# ---------------------------------------------------------------------------
def analyze_results(file: str,
                    print_examples: int = 0,
                    save_csv: bool = True,
                    csv_path: Optional[str] = None):
    """
    Analyze a results JSON produced by run_benchmark and print a compact summary.
    Usage: python dynamic_rag.py analyze_results --file path.json
    """
    with open(file, "r") as f:
        data = json.load(f)

    def clean_gt(s: str) -> str:
        s = (s or "").replace(",", "")
        s = re.sub(r'####\s*', '', s).strip()
        m = re.findall(r'([-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)', s)
        return m[-1] if m else s

    n = len(data)
    em = 0
    attempted = 0
    executed = 0
    final_tag = 0
    steps = []

    with_exec, without_exec = [], []
    mistakes = []

    for x in data:
        gt = clean_gt(x.get("ground_truth", ""))
        pred = str(x.get("predicted_answer") or "").strip()
        ok = (gt == pred)
        em += int(ok)
        attempted += int(bool(x.get("retrieval_attempted")))
        executed += int(bool(x.get("retrieval_executed")))
        final_tag += int(bool(x.get("predicted_answer_raw")))
        steps.append(x.get("total_steps", 0))
        (with_exec if x.get("retrieval_executed") else without_exec).append(x)
        if not ok:
            mistakes.append({
                "id": x.get("id"),
                "question": (x.get("question", "") or "")[:200],
                "gt": gt,
                "pred": pred,
                "retrieval_attempted": x.get("retrieval_attempted"),
                "retrieval_executed": x.get("retrieval_executed"),
                "pred_source": x.get("predicted_answer_source"),
            })

    def em_on(items):
        return sum(clean_gt(i.get("ground_truth", "")) == str(i.get("predicted_answer") or "").strip()
                   for i in items)

    em_exec = em_on(with_exec)
    em_noexec = em_on(without_exec)

    summary = {
        "n": n,
        "EM": em,
        "EM_pct": round(100*em/max(1, n), 1),
        "retrieval_attempted": attempted,
        "retrieval_attempted_pct": round(100*attempted/max(1, n), 1),
        "retrieval_executed": executed,
        "retrieval_executed_pct": round(100*executed/max(1, n), 1),
        "final_answer_declared_pct": round(100*final_tag/max(1, n), 1),
        "avg_steps": round(st.mean(steps), 2) if steps else 0.0,
        "EM_when_executed": em_exec,
        "EM_when_executed_pct": round(100*em_exec/max(1, len(with_exec)), 1) if with_exec else None,
        "EM_when_not_executed": em_noexec,
        "EM_when_not_executed_pct": round(100*em_noexec/max(1, len(without_exec)), 1) if without_exec else None,
    }

    print("\n=== Results Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if print_examples and mistakes:
        print("\n=== Example mistakes ===")
        for r in mistakes[:print_examples]:
            print(f"[id={r['id']}] pred={r['pred']} vs gt={r['gt']} | attempted={r['retrieval_attempted']} executed={r['retrieval_executed']} src={r.get('pred_source')}")
            print("Q:", r["question"])
            print("---")

    if save_csv and mistakes:
        out = csv_path
        if not out:
            base = os.path.splitext(os.path.basename(file))[0]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(os.path.dirname(file), "analysis")
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, f"{base}__mistakes_{stamp}.csv")
        with open(out, "w", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=list(mistakes[0].keys()))
            writer.writeheader()
            writer.writerows(mistakes)
        print(f"\nSaved mistakes CSV to: {out}")

    return summary


# ---------------------------------------------------------------------------
# Fire CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import fire
    fire.Fire({
        "run_benchmark": run_benchmark,
        "analyze_results": analyze_results,
    })
