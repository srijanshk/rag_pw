import csv
from datetime import datetime
import json
import os
import re
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple, List, Dict, Any

import fire


# -------- helpers for balanced \boxed{...} --------
def _balanced_after(s: str, start_idx: int) -> Optional[str]:
    """Return substring inside balanced braces starting at start_idx (right after '{')."""
    depth = 1
    i = start_idx
    out = []
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
    return None  # unbalanced / not found


def _extract_last_boxed_content(s: str) -> Optional[str]:
    """Find the LAST \\boxed{...} and return its balanced content (supports nested braces)."""
    tag = r'\boxed{'
    j = s.rfind(tag)
    if j == -1:
        return None
    return _balanced_after(s, j + len(tag))


def _normalize_answer(ans: str) -> str:
    """
    Normalize both predicted and ground-truth answers for robust EM.
    - Prefer the last LaTeX \frac{a}{b} anywhere in the string
    - Else extract balanced \boxed{...}
    - Strip $...$, commas, '####', spaces, trailing '.'
    - Canonicalize to reduced 'a/b' (or integer), decimals -> fraction
    - Otherwise return lowercased cleaned text
    """
    if ans is None:
        return ""
    s0 = str(ans).strip()
    if not s0:
        return ""

    # 0) prefer the LAST LaTeX fraction anywhere in the original string
    m_fracs = re.findall(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', s0)
    if m_fracs:
        num, den = map(int, m_fracs[-1])
        if den != 0:
            f = Fraction(num, den)
            return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    # 1) otherwise take balanced content from the LAST \boxed{...} if present
    boxed = _extract_last_boxed_content(s0)
    s = boxed if boxed is not None else s0

    # 2) cleanup
    s = s.replace("$", " ")
    s = re.sub(r'####\s*', '', s)
    s = s.replace(",", " ")
    s = s.replace("−", "-")  # normalize unicode minus
    s = re.sub(r'\s+', ' ', s).strip()
    if s.endswith('.'):
        s = s[:-1].strip()

    # 3) if there's now a LaTeX fraction, use it
    mf = re.search(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', s)
    if mf:
        num, den = int(mf.group(1)), int(mf.group(2))
        if den != 0:
            f = Fraction(num, den)
            return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        return s  # weird, keep as-is

    # 4) otherwise take the last numeric-like token: integer | a/b | decimal
    tokens = re.findall(r'[-+]?\d+(?:\s*/\s*[-+]?\d+|\.\d+)?', s)
    if tokens:
        t = tokens[-1].replace(' ', '')
        # decimal
        if re.match(r'^[-+]?\d+\.\d+$', t):
            try:
                f = Fraction(Decimal(t))
                return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
            except InvalidOperation:
                return t
        # fraction a/b
        if '/' in t:
            try:
                a, b = t.split('/')
                f = Fraction(int(a), int(b))
                return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
            except Exception:
                return t
        # integer
        try:
            return str(int(t))
        except Exception:
            return t

    # 5) non-numeric text answer (keep normalized text)
    return s.lower()


def _canon_math_text(s: str) -> str:
    """Light canonicalizer for LaTeX-ish math text (tuples/expressions).
    - Unwrap \boxed{...}
    - Replace \left/\right, unify \pi and π → pi
    - Convert \frac{a}{b} → a/b
    - Remove redundant spaces
    """
    if s is None:
        return ""
    t = str(s)
    # unwrap boxed
    t = re.sub(r"\\boxed{([^}]*)}", r"\1", t)
    # drop \left, \right
    t = re.sub(r"\\left|\\right", "", t)
    # frac → a/b
    t = re.sub(r"\\(?:d)?frac{([^}]*)}{([^}]*)}", r"\1/\2", t)
    # pi forms
    t = t.replace("π", "pi")
    t = re.sub(r"\\pi\b", "pi", t)
    # strip $ and commas
    t = t.replace("$", " ").replace(",", " ")
    # collapse ws
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_tuple_canonical(s: str) -> Optional[str]:
    """Return canonicalized tuple '(a,b)' if present, else None.
    Heuristic: take the LAST parenthesized group that contains a comma.
    """
    if not s:
        return None
    t = _canon_math_text(s)
    # find all paren groups
    matches = list(re.finditer(r"\(([^()]*)\)", t))
    for m in reversed(matches):
        inner = m.group(1)
        if "," in inner:
            a, b = inner.split(",", 1)
            a = _canon_math_text(a).replace(" ", "")
            b = _canon_math_text(b).replace(" ", "")
            return f"({a},{b})"
    return None


def _equals_em(gt: str, pred: str, pred_raw: Optional[str] = None) -> bool:
    ngt = _normalize_answer(gt)
    npred = _normalize_answer(pred)
    if ngt == npred:
        return True

    # Tuple-aware equality using raw strings if tuple forms exist
    gt_tuple = _extract_tuple_canonical(gt)
    pred_tuple = _extract_tuple_canonical(pred_raw or pred)
    if gt_tuple and pred_tuple:
        return gt_tuple == pred_tuple

    # Try numeric equivalence even if strings differ
    def to_fraction(x: str):
        try:
            if '/' in x:
                a, b = x.split('/')
                return Fraction(int(a), int(b))
            return Fraction(int(x), 1)
        except Exception:
            try:
                return Fraction(Decimal(x))
            except Exception:
                return None

    fx, fy = to_fraction(ngt), to_fraction(npred)
    return (fx is not None and fy is not None and fx == fy)


def _extract_full_query(x: dict) -> str:
    """
    Retrieve the most complete representation of the search query(ies).
    Falls back across several likely keys; JSON-encodes lists/dicts.
    """
    candidate_keys = [
        "full_query", "query", "search_query",
        "search_queries", "queries_used",
        "retrieval_trace", "search_trace"
    ]
    for k in candidate_keys:
        if k in x and x[k]:
            v = x[k]
            if isinstance(v, (list, dict)):
                return json.dumps(v, ensure_ascii=False)
            return str(v)
    return ""


def analyze_results(file: str,
                    print_examples: int = 3,
                    save_csv: bool = True,
                    csv_path: Optional[str] = None):
    """
    Analyze a results JSON produced by run_benchmark and print a compact summary.
    Usage: python analyze_results.py --file path/to/results.json
    """
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No data found in the file.")
        return {}

    n = len(data)

    # Counters
    em_total = 0
    em_with_retrieval = 0
    em_without_retrieval = 0

    retrieval_triggered = 0
    retrieval_attempted = 0
    retrieval_executed = 0

    count_with_retrieval = 0
    count_without_retrieval = 0

    steps = []
    mistakes = []

    # Retrieval count & doc-score metrics
    retrieval_counts: List[int] = []
    top_rerank_scores_correct: List[float] = []
    top_rerank_scores_incorrect: List[float] = []

    for x in data:
        gt_raw = x.get("ground_truth", "")
        pred_norm = x.get("predicted_answer") or ""
        pred_raw = x.get("predicted_answer_raw") or pred_norm
        ok = _equals_em(gt_raw, pred_norm, pred_raw)

        em_total += int(ok)
        steps.append(x.get("total_steps", 0) or 0)

        trig = bool(x.get("retrieval_triggered"))
        att = bool(x.get("retrieval_attempted"))
        exe = bool(x.get("retrieval_executed"))

        retrieval_triggered += int(trig)
        retrieval_attempted += int(att)
        retrieval_executed += int(exe)

        if exe:
            count_with_retrieval += 1
            if ok:
                em_with_retrieval += 1
        else:
            count_without_retrieval += 1
            if ok:
                em_without_retrieval += 1

        # Retrieval count
        rc = x.get("retrieval_count")
        if rc is None:
            rc = len(x.get("injections_made", []) or [])
        retrieval_counts.append(int(rc))

        # Top rerank score per example, if available
        top_score = None
        for inj in (x.get("injections_made") or []):
            for d in (inj.get("docs") or []):
                s = d.get("rerank_score")
                if isinstance(s, (int, float)):
                    top_score = s if top_score is None else max(top_score, s)
        if top_score is not None:
            (top_rerank_scores_correct if ok else top_rerank_scores_incorrect).append(float(top_score))

        if not ok:
            mistakes.append({
                "id": x.get("id"),
                "question": x.get("question", ""),
                "ground_truth_norm": _normalize_answer(gt_raw),
                "predicted_norm": _normalize_answer(pred_raw),
                "ground_truth_raw": gt_raw,
                "predicted_raw": pred_raw,
                "predicted_answer_source": x.get("predicted_answer_source"),
                "retrieval_triggered": trig,
                "retrieval_attempted": att,
                "retrieval_executed": exe,
                "queries_used": ", ".join(x.get("queries_used", []) or []),
                "full_query": _extract_full_query(x),          # NEW
                "full_reasoning": x.get("full_reasoning", ""), # optional, helpful
                "total_steps": x.get("total_steps"),
            })

    def safe_percent(num, den):
        return "N/A" if den == 0 else round(100 * num / den, 2)

    avg_rc_overall = round(sum(retrieval_counts) / n, 3) if n > 0 else 0.0
    avg_rc_retrieved = round((sum(c for c, x in zip(retrieval_counts, data) if x.get("retrieval_executed"))) / max(1, retrieval_executed), 3)

    summary = {
        "Total Problems": n,
        "Overall EM": f"{em_total}/{n}",
        "Overall EM (%)": safe_percent(em_total, n),
        "--- Retrieval Stats ---": "",
        "Retrieval Triggered (%)": safe_percent(retrieval_triggered, n),
        "Retrieval Attempted (%)": safe_percent(retrieval_attempted, n),
        "Retrieval Executed (%)": safe_percent(retrieval_executed, n),
        "EM with Retrieval": f"{em_with_retrieval}/{count_with_retrieval}",
        "EM with Retrieval (%)": safe_percent(em_with_retrieval, count_with_retrieval),
        "EM without Retrieval": f"{em_without_retrieval}/{count_without_retrieval}",
        "EM without Retrieval (%)": safe_percent(em_without_retrieval, count_without_retrieval),
        "--- Other ---": "",
        "Avg. Steps": round(sum(steps) / n, 2) if n > 0 else 0.0,
        "Avg. Retrieval Count (all)": avg_rc_overall,
        "Avg. Retrieval Count (exec only)": avg_rc_retrieved,
    }

    # Optional doc-score aggregates if present
    if top_rerank_scores_correct or top_rerank_scores_incorrect:
        def _avg(xs: List[float]) -> float:
            return round(sum(xs) / len(xs), 4) if xs else float('nan')
        summary.update({
            "Top-1 Rerank Score (correct)": _avg(top_rerank_scores_correct),
            "Top-1 Rerank Score (incorrect)": _avg(top_rerank_scores_incorrect),
        })

    print("\n✅ === Results Summary === ✅")
    for k, v in summary.items():
        if k.startswith("---"):
            print(f"\n{k}")
        else:
            print(f"{k:<25}: {v}")

    if print_examples and mistakes:
        print("\n❌ === Example Mistakes === ❌")
        for r in mistakes[:print_examples]:
            print(f"\n[ID: {r['id']}] Pred (norm): '{r['predicted_norm']}' | GT (norm): '{r['ground_truth_norm']}'")
            print(f"  Retrieval: trig={r['retrieval_triggered']}, att={r['retrieval_attempted']}, exec={r['retrieval_executed']} | Queries: {r['queries_used'] or 'None'}")
            print(f"  Full query: {r['full_query'][:2000] if r['full_query'] else '—'}")
            print(f"  Question: {r['question'][:200]}...")

    if save_csv:
        if not mistakes:
            print("\nNo mistakes found to save.")
            return summary

        out_path = csv_path
        if not out_path:
            base = os.path.splitext(os.path.basename(file))[0]
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            out_dir = os.path.join(os.path.dirname(file), "analysis")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base}__mistakes_{stamp}.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as wf:
            writer = csv.DictWriter(wf, fieldnames=list(mistakes[0].keys()))
            writer.writeheader()
            writer.writerows(mistakes)
        print(f"\n📝 Saved detailed mistake analysis to: {out_path}")

    return summary


if __name__ == "__main__":
    fire.Fire(analyze_results)
