import glob
import json
import os
import re
from collections import defaultdict
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple


# --- Normalization and equality helpers (aligned with analyze_result.py) ---
def _balanced_after(s: str, start_idx: int) -> Optional[str]:
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
    return None


def _extract_last_boxed_content(s: str) -> Optional[str]:
    tag = r'\boxed{'
    j = s.rfind(tag)
    if j == -1:
        return None
    return _balanced_after(s, j + len(tag))


def _normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    s0 = str(ans).strip()
    if not s0:
        return ""

    m_fracs = re.findall(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', s0)
    if m_fracs:
        num, den = map(int, m_fracs[-1])
        if den != 0:
            f = Fraction(num, den)
            return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    boxed = _extract_last_boxed_content(s0)
    s = boxed if boxed is not None else s0

    s = s.replace("$", " ")
    s = re.sub(r'####\s*', '', s)
    s = s.replace(",", " ")
    s = s.replace("−", "-")
    s = re.sub(r'\s+', ' ', s).strip()
    if s.endswith('.'):
        s = s[:-1].strip()

    mf = re.search(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', s)
    if mf:
        num, den = int(mf.group(1)), int(mf.group(2))
        if den != 0:
            f = Fraction(num, den)
            return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        return s

    tokens = re.findall(r'[-+]?\d+(?:\s*/\s*[-+]?\d+|\.\d+)?', s)
    if tokens:
        t = tokens[-1].replace(' ', '')
        if re.match(r'^[-+]?\d+\.\d+$', t):
            try:
                f = Fraction(Decimal(t))
                return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
            except InvalidOperation:
                return t
        if '/' in t:
            try:
                a, b = t.split('/')
                f = Fraction(int(a), int(b))
                return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
            except Exception:
                return t
        try:
            return str(int(t))
        except Exception:
            return t
    return s.lower()


def _canon_math_text(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    t = re.sub(r"\\boxed{([^}]*)}", r"\1", t)
    t = re.sub(r"\\left|\\right", "", t)
    t = re.sub(r"\\(?:d)?frac{([^}]*)}{([^}]*)}", r"\1/\2", t)
    t = t.replace("π", "pi")
    t = re.sub(r"\\pi\b", "pi", t)
    t = t.replace("$", " ").replace(",", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_tuple_canonical(s: str) -> Optional[str]:
    if not s:
        return None
    t = _canon_math_text(s)
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

    gt_tuple = _extract_tuple_canonical(gt)
    pred_tuple = _extract_tuple_canonical(pred_raw or pred)
    if gt_tuple and pred_tuple:
        return gt_tuple == pred_tuple

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


# --- File discovery helpers ---
def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


def find_latest_files() -> Dict[str, Dict[str, List[str]]]:
    """Return mapping dataset -> { mode -> [files] } where mode in {zero, static, dynamic-*}.
    dynamic-* includes corpus variants like 'dynamic-openmath' or 'dynamic-mathpile'.
    """
    out: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # Zero COT (no retrieval)
    for p in glob.glob('rag_pw/results/cot/*_COT_*.jsonl'):
        base = os.path.basename(p)
        dataset = base.split('_', 1)[0]
        out[dataset]['zero'].append(p)

    # Static RAG COT
    for p in glob.glob('rag_pw/results/static/*_STATIC_COT_*.jsonl'):
        base = os.path.basename(p)
        dataset = base.split('_', 1)[0]
        out[dataset]['static'].append(p)

    # Dynamic RAG COT summaries
    for p in glob.glob('rag_pw/results/dynamic/*_summary_*.json'):
        base = os.path.basename(p)
        # expected: <dataset>_<corpus>_summary_...
        parts = base.split('_')
        dataset = parts[0]
        corpus = parts[1] if len(parts) > 2 else 'unknown'
        out[dataset][f'dynamic-{corpus}'].append(p)

    # Dynamic RAG COT raw
    for p in glob.glob('rag_pw/results/dynamic/*_raw_*.json'):
        base = os.path.basename(p)
        # expected: <dataset>_<corpus>_raw_...
        parts = base.split('_')
        dataset = parts[0]
        corpus = parts[1] if len(parts) > 2 else 'unknown'
        out[dataset][f'dynamic-{corpus}-raw'].append(p)

    # sort by mtime desc
    for dataset, modes in out.items():
        for mode, files in modes.items():
            files.sort(key=_mtime, reverse=True)
    return out


# --- Metrics computation ---
def _safe_percent(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def eval_cot_like(path: str) -> Tuple[int, int]:
    """Evaluate EM for JSON array with keys prediction/answer."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    n = len(data)
    correct = 0
    for x in data:
        pred = x.get('predicted_answer') or x.get('prediction') or ''
        gt = x.get('ground_truth') or x.get('answer') or ''
        if _equals_em(gt, pred, pred):
            correct += 1
    return correct, n


def eval_dynamic_summary(path: str) -> Tuple[int, int, float]:
    """Evaluate EM and retrieval executed rate for dynamic summary list."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    n = len(data)
    correct = 0
    rexec = 0
    for x in data:
        pred = x.get('predicted_answer') or ''
        pred_raw = x.get('predicted_answer_raw') or pred
        gt = x.get('ground_truth') or ''
        if _equals_em(gt, pred, pred_raw):
            correct += 1
        if x.get('retrieval_executed'):
            rexec += 1
    return correct, n, _safe_percent(rexec, n)


def eval_dynamic_breakdown(path: str) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """Return (with_retrieval_correct, with_retrieval_total), (without_correct, without_total)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with_ret = [x for x in data if x.get('retrieval_executed')]
    without = [x for x in data if not x.get('retrieval_executed')]

    def score(xs):
        c = 0
        for x in xs:
            pred = x.get('predicted_answer') or ''
            pred_raw = x.get('predicted_answer_raw') or pred
            gt = x.get('ground_truth') or ''
            if _equals_em(gt, pred, pred_raw):
                c += 1
        return c, len(xs)

    return score(with_ret), score(without)


def build_markdown(note_path: str) -> None:
    groups = find_latest_files()

    lines: List[str] = []
    lines.append("# RAG COT Comparative Results\n")
    lines.append("This note summarizes EM accuracy for recent runs across modes: Zero COT, Static-RAG-COT, and Dynamic-RAG-COT (by corpus).\n")

    # per dataset
    for dataset in sorted(groups.keys()):
        lines.append(f"\n## Dataset: {dataset}\n")

        modes = groups[dataset]
        # Zero COT
        if modes.get('zero'):
            p = modes['zero'][0]
            c, n = eval_cot_like(p)
            acc = _safe_percent(c, n)
            lines.append(f"- Zero COT: {acc}% ({c}/{n}) — `{p}`")
        else:
            lines.append("- Zero COT: no recent file found")

        # Static
        if modes.get('static'):
            p = modes['static'][0]
            c, n = eval_cot_like(p)
            acc = _safe_percent(c, n)
            lines.append(f"- Static-RAG-COT: {acc}% ({c}/{n}) — `{p}`")
        else:
            lines.append("- Static-RAG-COT: no recent file found")

        # Dynamic variants
        dyn_keys = sorted([k for k in modes.keys() if k.startswith('dynamic-')])
        if dyn_keys:
            for dk in dyn_keys:
                p = modes[dk][0]
                c, n, rex = eval_dynamic_summary(p)
                acc = _safe_percent(c, n)
                corpus = dk.split('-', 1)[1]
                (cw, nw), (cwo, nwo) = eval_dynamic_breakdown(p)
                with_str = f"with: {round(100.0*cw/max(1,nw),2)}% ({cw}/{nw})"
                witho_str = f"without: {round(100.0*cwo/max(1,nwo),2)}% ({cwo}/{nwo})"

                # Compute deltas vs Zero/Static when available
                def _find_mode_acc(mode_key: str) -> Optional[float]:
                    if modes.get(mode_key):
                        _c, _n = eval_cot_like(modes[mode_key][0])
                        return _safe_percent(_c, _n)
                    return None

                base_zero = _find_mode_acc('zero')
                base_static = _find_mode_acc('static')
                def _delta_str(a: Optional[float]) -> str:
                    if a is None:
                        return ""
                    d = round(acc - a, 2)
                    sign = "+" if d >= 0 else ""
                    return f" (Δ vs base {sign}{d}pp)"

                kind = "summary"
                if corpus.endswith('-raw'):
                    kind = "raw"
                    corpus = corpus[:-4]

                delta_parts = []
                if base_zero is not None:
                    d = round(acc - base_zero, 2)
                    delta_parts.append(f"Δ vs Zero {d:+.2f}pp")
                if base_static is not None:
                    d = round(acc - base_static, 2)
                    delta_parts.append(f"Δ vs Static {d:+.2f}pp")
                delta_str = f" — {'; '.join(delta_parts)}" if delta_parts else ""

                lines.append(
                    f"- Dynamic-RAG-COT ({corpus}, {kind}): {acc}% ({c}/{n}), Retrieval Exec: {rex}% — {with_str}, {witho_str}{delta_str} — `{p}`"
                )
        else:
            lines.append("- Dynamic-RAG-COT: no recent file found")

    os.makedirs(os.path.dirname(note_path), exist_ok=True)
    with open(note_path, 'w', encoding='utf-8') as wf:
        wf.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    out_md = 'rag_pw/results/COMPARISON_NOTE.md'
    build_markdown(out_md)
    print(f"Wrote summary: {out_md}")
