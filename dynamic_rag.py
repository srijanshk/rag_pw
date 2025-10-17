from __future__ import annotations

from datetime import datetime
import os
from random import random
import re
import json
import glob
import logging
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager
import gc

import torch
import fire
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig
)

from FlagEmbedding import BGEM3FlagModel
from vllm import SamplingParams
from BGERetriever_v2 import BGERetriever

import wandb


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3,4")
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
# Header tokens: stop & sanitize
# ---------------------------------------------------------------------------

def sanitize_headers(text: str) -> str:
    """Remove Llama chat header tokens that sometimes leak into generations."""
    if not text:
        return text
    return text.strip()

# ---------------------------------------------------------------------------
# Custom stopping criteria: halt on ANY specified tag
# ---------------------------------------------------------------------------
class StopOnTag(StoppingCriteria):
    """Stop generation once a complete specified tag block is closed."""
    def __init__(self, tokenizer, stop_tags: List[str]):
        super().__init__()
        self.tok = tokenizer
        self.stop_sequences = [f"</{tag}>" for tag in stop_tags]
        self.buf = ""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.buf += self.tok.decode(input_ids[0, -1:], skip_special_tokens=True)
        # Keep buffer size manageable
        if len(self.buf) > 100:
            self.buf = self.buf[-100:]
        for seq in self.stop_sequences:
            if seq in self.buf:
                return True
        return False

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
# Query normalisation helpers (remove LaTeX, very long numbers)
# ---------------------------------------------------------------------------
_LATEX_CLEAN_RE = re.compile(r'\\boxed\{[^}]*\}|\\[a-zA-Z]+|[$]')
_LONG_NUM_RE = re.compile(r'\b\d{5,}\b')
_LATEX_TEXT_CMD_RE = re.compile(r"\\(?:text|mathrm|operatorname|textbf|textit|textrm|rm)\s*\{([^{}]*)\}")


def _extract_last_boxed_content(s: str) -> Optional[str]:
    """Return the balanced content of the last \boxed{...} block, if any."""
    tag = '\\boxed{'
    idx = s.rfind(tag)
    if idx == -1:
        return None
    depth = 1
    out = []
    i = idx + len(tag)
    while i < len(s):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return ''.join(out).strip()
        out.append(ch)
        i += 1
    return None


def strip_latex_text_wrappers(s: str) -> str:
    """Remove simple LaTeX text commands like \text{...} while preserving content."""
    if not s:
        return s
    prev = None
    out = s
    while prev != out:
        prev = out
        out = _LATEX_TEXT_CMD_RE.sub(lambda m: m.group(1).strip(), out)
    return out

def normalize_search_query(q: str) -> str:
    """
    Strip distracting LaTeX fragments and collapse spaces, but PRESERVE numbers
    and essential structure for a targeted search query.
    """
    # Remove LaTeX commands like \frac, \pmod, etc., and dollar signs
    q = re.sub(r'\\[a-zA-Z]+', ' ', q)
    q = q.replace('$', '')
    q = q.replace('{', '').replace('}', '')

    # Collapse multiple spaces into one and lowercase
    q = re.sub(r'\s+', ' ', q).strip()
    return q.lower()

SYSTEM_PROMPT = """
You are an expert mathematician.
Think step-by-step.
Write every reasoning step inside `<think> … </think>` blocks.

If you need to look up a formula, definition, or problems, you can use the <search> tool by writing a search query inside the <search> tag like this:
<search>your search query</search>

After retrieval, you may:
1. Use the information if helpful
2. Explicitly state "Retrieved information not helpful" and continue without it

After the search results are returned, continue your step-by-step thinking.

When you are ready to give the final answer, use the <answer> tag like this:
<answer>your final answer</answer> 
"""

INTEGRATE_PROMPT = """
Retrieved information:
<retrieved_knowledge>
{injected_text}
</retrieved_knowledge>

Strict rules for integration:
1. Extract the core mathematical principle (e.g., formula or theorem) in abstract terms—ignore specific numbers or examples.
2. State: 'Applying [principle name]: [abstract formula]'.
3. Map this to your problem's variables and show step-by-step application.
4. If the retrieval is not helpful (e.g., irrelevant or too specific), explicitly state 'Retrieval not used' and continue with your original reasoning.
5. Never copy numbers or solutions—adapt abstractly to avoid errors.
"""

VERIFICATION_PROMPT_TEMPLATE = """
CRITICAL VERIFICATION: Original problem: {problem}

Your tentative answer: <answer>{previous_answer}</answer>

Re-solve from scratch without retrieval:
1. Restate the problem.
2. Derive the solution step-by-step using alternative reasoning if possible.
3. Check for errors in logic, arithmetic, or assumptions.
4. If the answer matches, output: <answer>{previous_answer}</answer> and say 'VERIFIED'.
5. If there's an error, output: <answer>NEW_VALUE</answer> and say 'CORRECTION: [brief explanation]'.

Be thorough—show all calculations explicitly.
"""

CONSISTENCY_PROMPT = """
You have completed a solution using retrieved information.

Review the transcript above and double-check the final answer {previous_answer} without issuing any new <search> requests.

- If that answer is still correct, respond with exactly one <answer>{previous_answer}</answer>.
- If you discover an error, briefly correct the reasoning and finish with a single corrected <answer>VALUE</answer>.

Stick to the established <think>/<answer> format and do not retrieve new information.
"""

SUMMARY_SYSTEM_PROMPT = r"""
From the following evidence, extract a single, reusable method (definition, theorem, or algorithm) in a canonical, abstract form.

Your output must be a single, concise sentence following these rules:
- **Structure:** Start with the method's canonical name, followed by its formula in abstract variables, and end with any critical preconditions.
- **Content:** The formula must use abstract variables (e.g., a, b, n, x) not numbers from the problem.
- **Format:** Output only the single sentence. If no single canonical method can be extracted, output "UNHELPFUL".

**Good Example:**
The Binomial Theorem states that for a positive integer n, $(x+a)^n = \sum_{k=0}^{n} \binom{n}{k} x^k a^{n-k}$.

**Bad Example:**
To solve this, we used the binomial theorem to expand (2x+3)^4.
"""


def get_system_prompt(_: str = "") -> str:
    """Return the single unified prompt (bench argument kept for call‑site compatibility)."""
    return SYSTEM_PROMPT
# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------
def _latex_frac_to_str(s: str) -> Optional[str]:
    m = re.search(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None

def clean_answer(a: str) -> Optional[str]:
    """
    Extract a numeric answer (integer, decimal, or fraction) from an <answer> block.
    Robust to wrappers such as 'Final Answer:', '\boxed{...}', extra symbols, etc.
    Returns a canonical string like '123', '3.5', or '7/11', or None if not found.
    """
    if not a:
        return None
    boxed_inner = _extract_last_boxed_content(a)
    s = (boxed_inner if boxed_inner is not None else a).strip()

    # Remove leaked header tokens and repeated <answer> tags
    s = re.sub(r'<\|[^>]*\|>', '', s)
    s = re.sub(r'<answer>[^<]*</answer>\s*<answer>', '<answer>', s, flags=re.S)

    # Remove common verbal wrappers (case-insensitive)
    s = re.sub(r'(?i)\bfinal\s*answer\b\s*[:\-]?\s*', '', s)
    s = re.sub(r'(?i)\banswer\b\s*[:\-]?\s*', '', s)

    # Unwrap any \boxed{...} occurrences anywhere in the string
    s = re.sub(r'\\boxed\{([^}]*)\}', r'\1', s)

    # Remove simple LaTeX text wrappers like \text{...}
    s = strip_latex_text_wrappers(s)

    # Remove currency/percent and stray LaTeX $ markers and commas
    s = s.replace(',', '')
    s = s.replace('$', '')
    s = s.replace('%', '')
    s = s.replace('####', '')
    s = s.strip()

    # Prefer a LaTeX \frac{a}{b} anywhere in the string
    m_frac = re.search(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', s)
    if m_frac:
        return f"{m_frac.group(1)}/{m_frac.group(2)}"

    # Normalize whitespace once for pattern checks
    s_sp = re.sub(r'\s+', ' ', s).strip()

    # Simple numeric enclosed optionally by parentheses
    if re.fullmatch(r'\(?[-+]?\d+(?:/\d+|\.\d+)?\)?', s_sp):
        return s_sp.strip('()')

    # Tuples like (a,b) or (a,b,c)
    if re.fullmatch(r'\(([^()]*)\)', s_sp) and ',' in s_sp:
        inner = s_sp[1:-1]
        inner = inner.replace('\\pi', 'pi').replace('π', 'pi')
        inner = re.sub(r'\s+', '', inner)
        return f"({inner})"

    # Complex numbers a+bi or a-bi (allow fractions/decimals)
    if re.fullmatch(r'[-+]?\d+(?:/\d+|\.\d+)?\s*[+-]\s*\d+(?:/\d+|\.\d+)?i', s_sp):
        return s_sp.replace(' ', '')

    # Symbolic expressions with sqrt/pi/exponents/bases
    symbolic_markers = ('sqrt', '\\sqrt', 'π', '\\pi', '^', '_', 'sin', 'cos', 'tan', 'log', 'exp')
    if any(marker in s_sp for marker in symbolic_markers):
        s_sym = s_sp.replace('\\pi', 'pi').replace('π', 'pi')
        s_sym = re.sub(r'\s*\*\s*', '*', s_sym)
        return s_sym

    # Plain textual answers (names, labels, etc.)
    if s_sp and not re.search(r'\d', s_sp):
        return s_sp

    # Pull the last simple fraction OR decimal OR integer token
    tokens = re.findall(r'[-+]?\d+(?:/\d+|\.\d+)?', s_sp)
    if tokens:
        return tokens[-1]

    return None

# ---------------------------------------------------------------------------
# Final-answer extraction & normalization (robust)
# ---------------------------------------------------------------------------
def normalize_final_answer(ans_text: str) -> str:
    """
    Normalize a raw final-answer text to a comparison-friendly string.
    - Extracts a numeric from messy wrappers via clean_answer (with regex fallback).
    - Reduces fractions to lowest terms and collapses /1.
    - Trims trailing zeros in decimals and drops the dot if integer-valued.
    """
    s = clean_answer(ans_text or "")
    if not s:
        s = (ans_text or "").strip()
        s = re.sub(r'\\boxed\{([^}]*)\}', r'\1', s)
        s = strip_latex_text_wrappers(s)
        s = s.replace(',', '').replace('$', '').replace('%', '')
        m = re.search(r'([-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)', s)
        s = m.group(1) if m else s

    # If it's a fraction, reduce it
    if re.fullmatch(r"-?\d+/-?\d+", s):
        num_str, den_str = s.split('/')
        num, den = int(num_str), int(den_str)
        if den != 0:
            g = math.gcd(num, den)
            num //= g
            den //= g
            # Normalize sign to denominator > 0
            if den < 0:
                den = -den
                num = -num
            if den == 1:
                return str(num)
            return f"{num}/{den}"

    # If decimal, strip trailing zeros and drop the dot if integer-valued
    if re.fullmatch(r"-?\d+\.\d+", s):
        ip, fp = s.split('.', 1)
        fp = fp.rstrip('0')
        if fp == "":
            return ip
        return f"{ip}.{fp}"

    s = strip_latex_text_wrappers(s).strip()

    # Canonicalize purely textual answers by collapsing whitespace and lowercasing
    if s and not re.search(r'\d', s):
        s = re.sub(r'\s+', ' ', s).strip().lower()

    return s

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def generate_with_stop(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stop_tags: List[str],
    temperature: float = 0.1,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
    stopping_criteria = StoppingCriteriaList([StopOnTag(tokenizer, stop_tags)])

    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=False,
            )
        gen = tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return gen.strip()
    finally:
        # Aggressive cleanup of per-call tensors
        try:
            del inputs
        except Exception:
            pass
        try:
            del out
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



def generate_with_stop_vllm(
    engine, prompt: str, max_new_tokens: int, stop_tags, temperature: float = 0.1, seed: int = 123
) -> str:
    stop = [f"</{t}>" for t in stop_tags]           # e.g., ["</search>", "</answer>"]
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.05,
        stop=stop,
        seed=seed,
    )
    out = engine.generate([prompt], params)
    return out[0].outputs[0].text.strip()


def parse_tag(text: str, tag: str) -> Tuple[Optional[str], str]:
    """Extracts content from the first complete tag block, e.g., <tag>content</tag>."""
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    # Regex to grab the first complete block non-greedily
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    if not m:
        return None, text
    content = (m.group(1) or "").strip()
    cleaned_text = (text[:m.start()] + text[m.end():]).strip()
    return content, cleaned_text

# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------
def get_doc_text(d: dict) -> str:
    return d.get("solution_chunk") or d.get("solution") or d.get("text", "")


# ---------------------------------------------------------------------------
# Context summarization (method-only injection)
# ---------------------------------------------------------------------------

def summarize_context_with_llm(
    model,
    tokenizer,
    query: str,
    context_str: str,
    problem: Optional[str] = None,
    reasoning_so_far: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:

    # Short-circuit if nothing useful to summarize
    if not context_str or not context_str.strip():
        return ""

    # Local helpers -------------------------------------------------------
    def _sanitize_reasoning(s: str) -> str:
        # Drop leaked answers or control tags that could steer generation
        s = re.sub(r"<answer>.*?</answer>", "", s, flags=re.S | re.I)
        s = re.sub(r"(?i)final\s*answer\s*[:\-]?", "", s)
        s = re.sub(r"<search>.*?</search>", "", s, flags=re.S | re.I)
        s = s.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
        return s.strip()

    def _postprocess(summary: str) -> str:
        # Enforce no answers and avoid repeated whitespace; NO word-cap truncation
        summary = re.sub(r"<answer>.*?</answer>", "", summary, flags=re.S | re.I)
        summary = re.sub(r"(?i)final\s*answer\s*[:\-]?", "", summary)
        summary = re.sub(r"[ \t]+\n", "\n", summary)
        return summary.strip()

    problem_snip = problem or ""
    reasoning_snip = _sanitize_reasoning(reasoning_so_far or "")
    evidence_snip = context_str

    user = (
        f"<query>\n{query}\n</query>\n\n"
        f"<evidence>\n{evidence_snip}\n</evidence>"
    )

    prompt = build_llama_prompt(SUMMARY_SYSTEM_PROMPT, user)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.02,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return _postprocess(raw)
    except Exception:
        return ""

def summarize_context_with_vllm(
    model,                      # vLLM LLM engine
    query: str,
    context_str: str,
    problem: Optional[str] = None,
    reasoning_so_far: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """
    Summarize retrieved context into a one-sentence 'method card' using vLLM.
    Returns a short summary or '' on failure. Matches the original function's API.
    """
    import re
    from vllm import SamplingParams

    # Short-circuit if nothing useful to summarize
    if not context_str or not context_str.strip():
        return ""

    # ---- helpers ---------------------------------------------------------
    def _sanitize_reasoning(s: str) -> str:
        s = re.sub(r"<answer>.*?</answer>", "", s, flags=re.S | re.I)
        s = re.sub(r"(?i)\bfinal\s*answer\b\s*[:\-]?", "", s)
        s = re.sub(r"<search>.*?</search>", "", s, flags=re.S | re.I)
        s = s.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
        return s.strip()

    def _postprocess(summary: str) -> str:
        summary = re.sub(r"<answer>.*?</answer>", "", summary, flags=re.S | re.I)
        summary = re.sub(r"(?i)\bfinal\s*answer\b\s*[:\-]?", "", summary)
        summary = re.sub(r"[ \t]+\n", "\n", summary)
        return summary.strip()

    problem_snip = problem or ""
    reasoning_snip = _sanitize_reasoning(reasoning_so_far or "")

    user = (
        f"<query>\n{query}\n</query>\n\n"
        f"<evidence>\n{context_str}\n</evidence>"
    )
    prompt = build_llama_prompt(SUMMARY_SYSTEM_PROMPT, user)

    try:
        params = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=0.9,
            repetition_penalty=1.02,
            # # light safety stops; harmless if never produced
            # stop=["<|eot_id|>", "</answer>"],
            seed=123,
        )
        out = model.generate([prompt], params)
        raw = (out[0].outputs[0].text or "").strip()
        return _postprocess(raw)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Core solve loop
# ---------------------------------------------------------------------------
def solve(problem: str,
          retriever,
          engine,
          tokenizer,
          bench: str,
          k_dense: int,
          k_final: int,
          max_tool_calls: int = 3,
          tool_gen_tokens: int = 512,
          answer_gen_tokens: int = 2048,
          seed: int = 123,
          injection_mode: str = "summary",
          ) -> Dict:

    sys_prompt = get_system_prompt(bench)
    prompt = build_llama_prompt(sys_prompt, problem)

    transcript: List[str] = []
    trace: List[dict] = []

    retrieval_triggered = False
    retrieval_executed = False
    answer_content: Optional[str] = None
    verification_needed = False
    verification_requested = False
    verification_used = False
    verification_blocks = 0
    verification_successes = 0
    verification_failures = 0
    consistency_used = False
    consistency_changed = False
    consistency_raw: Optional[str] = None

    def run_verification_step(current_answer: str) -> Tuple[str, bool]:
        nonlocal prompt, verification_used, verification_blocks, verification_successes, verification_failures

        prev_answer = (current_answer or "").strip()
        retry_prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            problem=problem,
            previous_answer=prev_answer.replace('{', '{{').replace('}', '}}') or "the current answer"
        )

        for attempt in range(2):
            verification_used = True
            verification_blocks += 1

            prompt = append_block(prompt, "user", retry_prompt)
            transcript.append(
                "<verification_request attempt=\"{attempt}\">\n".format(attempt=attempt + 1)
                + retry_prompt.strip()
                + "\n</verification_request>"
            )

            verify_gen = generate_with_stop(
                model=engine,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=answer_gen_tokens,
                stop_tags=["answer"],
                temperature=0,
            )
            verify_gen = sanitize_headers(verify_gen)

            transcript.append(
                "<verification_response attempt=\"{attempt}\">\n".format(attempt=attempt + 1)
                + verify_gen
                + "\n</verification_response>"
            )
            prompt = append_block(prompt, "assistant", verify_gen)

            ans_verify, _ = parse_tag(verify_gen, "answer")
            candidate = (ans_verify or "").strip() if ans_verify is not None else ""
            lower_text = verify_gen.lower()
            has_digits = bool(re.search(r"\d", candidate))

            if candidate:
                if candidate == prev_answer:
                    if "verified" in lower_text:
                        verification_successes += 1
                        return candidate, True
                    else:
                        retry_prompt = (
                            "You did not explicitly show a full recomputation or state 'Verified'. "
                            f"The target answer remains <answer>{prev_answer}</answer>. "
                            "Re-derive the solution step-by-step, display all arithmetic, and conclude with 'Verified' if it matches."
                        )
                        continue
                else:
                    if "correction" in lower_text or has_digits:
                        verification_successes += 1
                        return candidate, True
                    retry_prompt = (
                        f"You proposed a different value <answer>{candidate}</answer> without marking it as a correction. "
                        "If the answer changes, explicitly say 'Correction', show the recalculated steps, and provide the corrected <answer>."
                    )
                    prev_answer = candidate
                    continue

            retry_prompt = (
                "Your verification response was incomplete. Recompute from scratch, show the steps, and finish with a single <answer>."
            )

        verification_failures += 1
        return current_answer, False

    for turn in range(max_tool_calls):
        gen = generate_with_stop(
            model=engine,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=tool_gen_tokens,
            stop_tags=["search", "answer"],
            temperature=0,
        )
        gen = sanitize_headers(gen)
        transcript.append(gen)

        ans, cleaned_text_ans = parse_tag(gen, "answer")
        if ans is not None:
            inner = (ans or "").strip()
            if inner:
                prompt = append_block(prompt, "assistant", gen)
                answer_content = inner
                if verification_needed and not verification_requested:
                    answer_content, verify_ok = run_verification_step(answer_content)
                    verification_requested = True
                    verification_needed = False
                    if not verify_ok:
                        # mark for potential follow-up by consistency check
                        pass
                else:
                    verification_needed = False
                break
            else:
                prompt = append_block(
                    prompt,
                    "user",
                    "Format error: The <answer> tag was empty. Please provide a final answer inside the tag.",
                )
                continue

        # Otherwise, check for a search request
        search_query, cleaned_text_search = parse_tag(gen, "search")
        if search_query:
            retrieval_triggered = True
            retrieval_executed = True
            query_norm = normalize_search_query(search_query)
            docs = []
            if retriever:
                docs = retriever.search_and_rerank(
                    query_norm,
                    top_k=k_dense,
                    top_k_final=k_final,
                )
            context_str = format_passages(docs)
            # Minimal per-doc stats for analysis (avoid injecting raw text here)
            docs_stats = []
            try:
                for d in docs:
                    docs_stats.append({
                        "id": d.get("id"),
                        "dense_score": d.get("dense_score"),
                        "rerank_score": d.get("rerank_score"),
                        "problem_from": d.get("problem_from"),
                        "row_id": d.get("row_id"),
                        "chunk_id": d.get("chunk_id"),
                    })
            except Exception:
                docs_stats = []
            if injection_mode == "summary":
                recent_reasoning = "\n\n".join(transcript[-3:]) if transcript else ""
                method_only = summarize_context_with_llm(
                    engine,
                    tokenizer,
                    query_norm,
                    context_str,
                    problem=problem,
                    reasoning_so_far=recent_reasoning,
                )
                inject_text = (method_only or "").strip()
            else:
                inject_text = (context_str or "").strip()

            inject_display = inject_text

            transcript.append(
                "<retrieval>\n"
                f"<search>{search_query}</search>\n"
                "<context>\n"
                f"{inject_display}\n"
                "</context>\n"
                "</retrieval>"
            )

            trace.append({
                "query_raw": search_query,
                "query": query_norm,
                "ctx_raw": context_str,
                "ctx_injected": inject_text,
                "ctx_injected_preview": inject_display,
                "num_docs_found": len(docs),
                "injection_mode": injection_mode,
                "docs": docs_stats,
            })

            verification_needed = True
            verification_requested = False

            if inject_text:
                injection_prompt = INTEGRATE_PROMPT.format(
                    injected_text=inject_text,
                    problem=problem,
                )
                prompt = append_block(prompt, "user", injection_prompt)
            else:
                prompt = append_block(
                    prompt,
                    "user",
                    f"<problem>\n{problem}\n</problem>\n",
                )
            continue

        # No <answer> and no <search>: carry the text forward and try again
        if gen.strip():
            prompt = append_block(prompt, "assistant", gen)

    if answer_content is None:
        final_shot_prompt = append_block(
            prompt,
            "user",
            (
                "Your reasoning is stuck. Please output a final answer directly. "
                "Output exactly one line in the form\n"
                "<answer>VALUE</answer>\n"
                "No other text. Do NOT include <think> or <search>. Ensure VALUE matches the required form stated in <interpret>."
            ),
        )
        final_gen = generate_with_stop(
            model=engine,
            tokenizer=tokenizer,
            prompt=final_shot_prompt,
            max_new_tokens=answer_gen_tokens,
            stop_tags=["answer"],
            temperature=0,
        )
        final_clean = sanitize_headers(final_gen)
        transcript.append(f"<forced_final_generation>\n{final_clean}\n</forced_final_generation>")
        ans2, _ = parse_tag(final_clean, "answer")
        if ans2 is not None:
            inner2 = (ans2 or "").strip()
            if inner2:
                answer_content = inner2
                prompt = append_block(prompt, "assistant", final_clean)

    if retrieval_executed and answer_content and verification_blocks == 0:
        answer_content, _ = run_verification_step(answer_content)
        verification_requested = True

    queries_used = [t.get("query", t.get("query_raw")) for t in trace]

    # Self-consistency double-check when retrieval informed the solve
    if retrieval_executed and answer_content:
        prev_for_prompt = answer_content.strip()
        prev_for_prompt = prev_for_prompt.replace('{', '{{').replace('}', '}}')
        consistency_instruction = CONSISTENCY_PROMPT.format(
            previous_answer=prev_for_prompt or "the current final answer"
        )

        consistency_prompt = append_block(
            prompt,
            "user",
            consistency_instruction,
        )

        consistency_gen = generate_with_stop(
            model=engine,
            tokenizer=tokenizer,
            prompt=consistency_prompt,
            max_new_tokens=answer_gen_tokens,
            stop_tags=["answer"],
            temperature=0,
        )
        consistency_gen = sanitize_headers(consistency_gen)

        transcript.append(
            "<consistency_check>\n"
            f"{consistency_gen}\n"
            "</consistency_check>"
        )

        ans_consistency, _ = parse_tag(consistency_gen, "answer")
        if ans_consistency is not None:
            candidate = (ans_consistency or "").strip()
            if candidate:
                consistency_used = True
                consistency_raw = candidate
                prev_norm = normalize_final_answer(answer_content)
                new_norm = normalize_final_answer(candidate)
                if new_norm and new_norm != prev_norm:
                    # Only upgrade the answer if the consistency check explicitly signals a revision
                    lower_case_gen = consistency_gen.lower()
                    if any(keyword in lower_case_gen for keyword in ("correct", "mistake", "error", "fix")):
                        answer_content = candidate
                        consistency_changed = True

    # Rebuild reasoning/output (may include consistency block)
    full_reasoning_out = "\n\n---\n".join(transcript)
    if queries_used:
        full_reasoning_out += "\n\n---\nSearch queries used:\n" + "\n".join(f"- {q}" for q in queries_used)

    # Derive final predicted answer
    predicted_answer = None
    pred_source = None
    raw_pred_text = None

    if answer_content is not None:
        raw_pred_text = answer_content
        predicted_answer = normalize_final_answer(answer_content)
        pred_source = "answer_tag"
    else:
        # last-resort scrape from transcript
        nums = re.findall(r'([-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)', full_reasoning_out)
        if nums:
            raw_pred_text = nums[-1]
            predicted_answer = normalize_final_answer(raw_pred_text)
            pred_source = "numeric_fallback"

    return {
        "predicted_answer": predicted_answer,
        "predicted_answer_raw": raw_pred_text,
        "predicted_answer_source": pred_source,
        "full_reasoning": full_reasoning_out,
        "queries_used": queries_used,
        "retrieval_attempted": retrieval_executed,
        "retrieval_executed": retrieval_executed,
        "retrieval_triggered": retrieval_triggered,
        "injections_made": trace,
        "retrieval_count": len(trace),
        "total_steps": len(transcript),
        "verification_used": verification_used,
        "verification_success": verification_successes,
        "verification_failure": verification_failures,
        "verify_blocks": verification_blocks,
        "consistency_used": consistency_used,
        "consistency_changed": consistency_changed,
        "consistency_answer_raw": consistency_raw,
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
    k_dense: int = 200,
    k_final: int = 5,
    tool_gen_tokens: int = 512,
    answer_gen_tokens: int = 2048,
    max_tool_calls: int = 3,
    wandb_project: Optional[str] = None,
    wandb_run: Optional[str] = None,
    out_path: Optional[str] = None,
    seed: int = 42,
    quantize_4bit: bool = False,
    injection_mode: str = "summary",
):
    # dataset discovery ------------------------------------------------------
    if dataset_path is None:
        hf_map = {
            "math": "hendrycks/competition_math",
            "gsm8k": "gsm8k",
            "math500": "HuggingFaceH4/MATH-500",
            "math-500": "HuggingFaceH4/MATH-500",
        }
        root = f"./data/benchmarks/{benchmark}"
        hf = hf_map.get(benchmark)
    #     for p in [f"{root}/test.jsonl", root, hf]:
    #         if os.path.exists(p) or not p.startswith("."):
    #             dataset_path = p
    #             break
    # ds = load_split_any(dataset_path, "test")
        candidates = [f"{root}/test.jsonl", root]
        if hf:
            candidates.append(hf)

        dataset_path = None
        for p in candidates:
            if (p.startswith(".") and os.path.exists(p)) or (not p.startswith(".")):
                dataset_path = p
                break

        if dataset_path is None:
            raise ValueError(
                f"Couldn't resolve dataset for benchmark='{benchmark}'. "
                f"Pass --dataset_path explicitly."
            )

    ds = load_split_any(dataset_path, "test")
    if num_samples and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    try:
        import random as _r
        _r.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # model & tokenizer ------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    tok.pad_token = tok.eos_token

    # engine = LLM(
    #         model=model_name,
    #         tensor_parallel_size=2,             
    #         dtype="float16",                   
    #         max_model_len=64_000,     
    #         enforce_eager=True,           
    #         compilation_config={"use_cudagraph": False},        
    #         enable_chunked_prefill=False,                    
    #     )

    qcfg = None
    if quantize_4bit:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    engine = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=qcfg,
        low_cpu_mem_usage=True,
        offload_folder="/tmp/offload",  
        offload_state_dict=True,
    )
    engine.eval()

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
            max_tool_calls=max_tool_calls,
        ))
    else:
        console.print("[grey62]W&B disabled[/grey62]")

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
            engine=engine,
            tokenizer=tok,
            bench=benchmark,
            k_dense=k_dense,
            k_final=k_final,
            tool_gen_tokens=tool_gen_tokens,
            answer_gen_tokens=answer_gen_tokens,
            max_tool_calls=max_tool_calls,
            seed=seed,
            injection_mode=injection_mode,
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_path if out_path else f"./results/{benchmark}_streamrag_{timestamp}.json"
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
    fire.Fire(run_benchmark)
