"""Utilities for evaluating RAG vs. CoT runs on gsm8k and math500.

This module centralises data loading, answer normalisation, and metric
computation so that research notebooks can focus on analysis rather than
boilerplate.  It supports:

- Baseline chain-of-thought runs (no retrieval)
- Static RAG-CoT runs
- Adaptive / dynamic RAG runs (v2 directory)

The helpers expose pandas DataFrames annotated with benchmark metadata
(e.g. subject, difficulty), normalised answers, and retrieval telemetry
where available.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyze_result import _equals_em, _normalize_answer  # reuse existing logic

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

GSM_CATEGORY_RULES: List[Tuple[str, Iterable[str]]] = [
    ("Money & Finance", ["dollar", "$", "cents", "cost", "price", "profit", "sale", "ticket"]),
    ("Rates & Work", ["mph", "mile", "per hour", "speed", "rate", "per minute", "per day", "work", "together", "fill", "drain"]),
    ("Fractions & Ratios", ["fraction", "ratio", "percent", "percentage", "proportion", "share", "split", "divide equally"]),
    ("Counting & Combinatorics", ["ways", "arrangements", "combinations", "permutations", "orderings", "seating"]),
    ("Geometry & Measurement", ["area", "perimeter", "volume", "circle", "triangle", "rectangle", "square", "yard", "meter", "feet", "inch"]),
    ("Time & Scheduling", ["minutes", "hours", "clock", "time", "later", "earlier", "calendar", "day", "week"]),
    ("Logic & Number Puzzles", ["numbers", "greater", "less", "difference", "sum", "product", "twice", "triple", "age", "years old"]),
]


@dataclass(frozen=True)
class RunSpec:
    """Descriptor for a single evaluation file."""

    label: str
    dataset: str  # 'gsm8k' or 'math500'
    mode: str  # 'cot', 'static', 'dynamic'
    path: Path
    corpus: Optional[str] = None  # e.g. 'mathpile', 'openmath'
    context: Optional[str] = None  # e.g. 'summary', 'raw'
    notes: Optional[str] = None

    def short_name(self) -> str:
        bits = [self.mode]
        if self.corpus:
            bits.append(self.corpus)
        if self.context:
            bits.append(self.context)
        return "-".join(bits)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def _clean_subject(s: str) -> str:
    s = (s or "").replace("_", " ").strip()
    return s.title() if s else "Unknown"


@lru_cache(maxsize=None)
def load_dataset_frame(dataset: str) -> pd.DataFrame:
    """Load benchmark questions/answers plus metadata.

    For gsm8k we add heuristic categories and difficulty bands based on
    question length quantiles.  For math500 we surface the provided
    subject and difficulty level.
    """

    dataset = dataset.lower()
    if dataset not in {"gsm8k", "math500"}:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    if dataset == "gsm8k":
        from datasets import load_dataset  # lazy import to avoid heavy cost if unused

        ds = load_dataset("gsm8k", "main", split="test")
        df = ds.to_pandas()
        df = df.rename(columns={"answer": "ground_truth"})
        df.insert(0, "idx", np.arange(len(df)))

        # Heuristic categories
        lowered = df["question"].str.lower()
        categories = []
        for text in lowered:
            cat = "Other"
            for label, keywords in GSM_CATEGORY_RULES:
                if any(kw in text for kw in keywords):
                    cat = label
                    break
            categories.append(cat)
        df["category"] = categories

        lengths = lowered.str.len()
        q1, q2 = np.quantile(lengths, [0.33, 0.66])

        def _band(n: int) -> str:
            if n <= q1:
                return "Short"
            if n <= q2:
                return "Medium"
            return "Long"

        df["difficulty"] = lengths.apply(lambda n: _band(int(n)))
        return df[["idx", "question", "ground_truth", "category", "difficulty"]]

    # math500 branch
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    df = ds.to_pandas()
    df = df.rename(columns={"answer": "ground_truth", "problem": "question"})
    df.insert(0, "idx", np.arange(len(df)))
    df["category"] = df["subject"].apply(_clean_subject)
    level_map = {
        1: "Level 1",
        2: "Level 2",
        3: "Level 3",
        4: "Level 4",
        5: "Level 5",
    }
    df["difficulty"] = df["level"].map(level_map).fillna("Unknown")
    return df[["idx", "question", "ground_truth", "category", "difficulty", "subject", "level", "unique_id"]]


# ---------------------------------------------------------------------------
# Result file loading & normalisation
# ---------------------------------------------------------------------------


def _load_json_flex(path: Path) -> List[dict]:
    """Load either JSON array or JSON lines file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    items = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            items.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return items


def _ensure_len(items: List[dict], expected: int, spec: RunSpec) -> List[dict]:
    if len(items) != expected:
        raise ValueError(
            f"Length mismatch for {spec.label}: got {len(items)}, expected {expected}"
        )
    return items


def _normalise_predictions(raw_items: List[dict], spec: RunSpec) -> pd.DataFrame:
    """Convert raw prediction dicts into a tabular form with uniform columns."""

    df = pd.DataFrame(raw_items)
    df = df.reset_index().rename(columns={"index": "idx"})

    # Standardise answer columns
    if spec.mode in {"cot", "static", "llm"}:
        df = df.rename(columns={"prediction": "predicted_answer", "answer": "ground_truth_raw"})
        df["predicted_answer_raw"] = df.get("raw")
    else:
        df = df.rename(columns={"ground_truth": "ground_truth_raw"})

    if "ground_truth_raw" not in df.columns:
        df["ground_truth_raw"] = None

    if "predicted_answer" not in df.columns:
        df["predicted_answer"] = df.get("predicted_answer_raw")

    # Retrieval telemetry sane defaults
    for col, default in [("retrieval_triggered", False), ("retrieval_executed", False), ("retrieval_attempted", False)]:
        if col in df.columns:
            df[col] = df[col].fillna(default).astype(bool)
        else:
            df[col] = default

    if "retrieval_count" in df.columns:
        df["retrieval_count"] = df["retrieval_count"].fillna(0).astype(float)
    else:
        df["retrieval_count"] = 0.0

    return df


def load_run_dataframe(spec: RunSpec) -> pd.DataFrame:
    """Load predictions for a run and align them with the reference dataset."""

    ds = load_dataset_frame(spec.dataset)
    raw_items = _load_json_flex(spec.path)
    raw_items = _ensure_len(raw_items, len(ds), spec)
    pred_df = _normalise_predictions(raw_items, spec)

    merged = ds.merge(pred_df, on="idx", how="left", suffixes=("", "_pred"))
    merged["ground_truth_norm"] = merged["ground_truth"].apply(_normalize_answer)
    merged["prediction_norm"] = merged["predicted_answer"].apply(_normalize_answer)
    merged["is_correct"] = merged.apply(
        lambda row: _equals_em(row["ground_truth"], row["predicted_answer"], row.get("predicted_answer_raw")),
        axis=1,
    )

    # Provide reasoning text column for convenience
    reasoning_cols = [
        "full_reasoning",
        "raw",
        "predicted_answer_raw",
    ]
    for col in reasoning_cols:
        if col in merged and merged[col].notna().any():
            merged["reasoning"] = merged[col]
            break
    else:
        merged["reasoning"] = None

    merged["run_label"] = spec.label
    merged["mode"] = spec.mode
    merged["corpus"] = spec.corpus
    merged["context_type"] = spec.context
    return merged


# ---------------------------------------------------------------------------
# Run discovery helpers
# ---------------------------------------------------------------------------


def _latest_in_dir(pattern: str) -> Optional[Path]:
    paths = sorted(Path(p) for p in RESULTS_DIR.glob(pattern))
    return paths[-1] if paths else None


def discover_default_specs() -> Dict[str, dict]:
    """Return a dict keyed by mode with dataset -> RunSpec mappings.

    For dynamic runs the value is itself a dict mapping each corpus/context
    variant to its RunSpec so notebooks can iterate over available options.
    """

    specs: Dict[str, dict] = {"llm": {}, "cot": {}, "static": {}, "dynamic": {}}

    # Baseline CoT (no retrieval)
    gsm_cot = _latest_in_dir("cot/gsm_COT_*.jsonl")
    if gsm_cot:
        specs["cot"]["gsm8k"] = RunSpec("CoT (gsm8k)", "gsm8k", "cot", gsm_cot)
    math_cot = _latest_in_dir("cot/math500_COT_*.jsonl")
    if math_cot:
        specs["cot"]["math500"] = RunSpec("CoT (math500)", "math500", "cot", math_cot)
    
    # LLM (no retrieval)
    gsm_llm = _latest_in_dir("llm/gsm_LLM_*.jsonl")
    if gsm_llm:
        specs["llm"]["gsm8k"] = RunSpec("LLM (gsm8k)", "gsm8k", "llm", gsm_llm)
    math_llm = _latest_in_dir("llm/math500_LLM_*.jsonl")
    if math_llm:
        specs["llm"]["math500"] = RunSpec("LLM (math500)", "math500", "llm", math_llm)

    # Static RAG-CoT
    gsm_static = _latest_in_dir("static/gsm_STATIC_COT_*.jsonl")
    if gsm_static:
        specs["static"]["gsm8k"] = RunSpec("Static RAG-CoT (gsm8k)", "gsm8k", "static", gsm_static)
    math_static = _latest_in_dir("static/math500_STATIC_COT_*.jsonl")
    if math_static:
        specs["static"]["math500"] = RunSpec("Static RAG-CoT (math500)", "math500", "static", math_static)

    # Adaptive dynamic runs (v2 only)
    dyn_specs: Dict[Tuple[str, str, str], Tuple[str, Path]] = {}
    pattern = re.compile(
        r"(gsm8k|math500)_(mathpile|openmath)_(summary|raw)_([0-9]{8}_[0-9]{4})"
    )
    dyn_dir = RESULTS_DIR / "dynamic_v2"
    if dyn_dir.exists():
        for path in dyn_dir.glob("*.json"):
            m = pattern.search(path.name)
            if not m:
                continue
            key = (m.group(1), m.group(2), m.group(3))
            stamp = m.group(4)
            prev = dyn_specs.get(key)
            if not prev or stamp > prev[0]:
                dyn_specs[key] = (stamp, path)

    for (dataset, corpus, context), (_, path) in dyn_specs.items():
        label = f"Dynamic {corpus.title()} ({context}) [{dataset}]"
        dataset_bucket = specs["dynamic"].setdefault(dataset, {})
        dataset_bucket[f"{corpus}_{context}"] = RunSpec(
            label=label,
            dataset=dataset,
            mode="dynamic",
            path=path,
            corpus=corpus,
            context=context,
        )

    return specs


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_accuracy(df: pd.DataFrame) -> float:
    return float(df["is_correct"].mean()) if not df.empty else float("nan")


def accuracy_breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not present in dataframe.")
    grouped = (
        df.groupby(col)["is_correct"].agg(["mean", "count", "sum"]).rename(columns={
            "mean": "accuracy",
            "count": "n",
            "sum": "n_correct",
        }).reset_index()
    )
    grouped["accuracy"] = grouped["accuracy"].round(4)
    return grouped.sort_values("accuracy", ascending=False)


def retrieval_stats(df: pd.DataFrame) -> pd.Series:
    if "retrieval_executed" not in df.columns:
        return pd.Series(dtype=float)
    executed = df["retrieval_executed"].astype(bool)
    counts = df.get("retrieval_count", pd.Series([0] * len(df)))
    return pd.Series({
        "retrieval_rate": executed.mean(),
        "avg_retrievals_all": counts.mean(),
        "avg_retrievals_when_used": counts[executed].mean() if executed.any() else 0.0,
    })


def merge_runs(df_a: pd.DataFrame, df_b: pd.DataFrame, suffixes: Tuple[str, str] = ("_a", "_b")) -> pd.DataFrame:
    cols = ["idx", "question", "ground_truth", "category", "difficulty"]
    base = df_a[cols]
    merged = base.merge(
        df_a[["idx", "predicted_answer", "prediction_norm", "is_correct", "reasoning"]].rename(
            columns={
                "predicted_answer": f"predicted{suffixes[0]}",
                "prediction_norm": f"prediction_norm{suffixes[0]}",
                "is_correct": f"is_correct{suffixes[0]}",
                "reasoning": f"reasoning{suffixes[0]}",
            }
        ),
        on="idx",
        how="left",
    )
    merged = merged.merge(
        df_b[["idx", "predicted_answer", "prediction_norm", "is_correct", "reasoning"]].rename(
            columns={
                "predicted_answer": f"predicted{suffixes[1]}",
                "prediction_norm": f"prediction_norm{suffixes[1]}",
                "is_correct": f"is_correct{suffixes[1]}",
                "reasoning": f"reasoning{suffixes[1]}",
            }
        ),
        on="idx",
        how="left",
    )
    return merged


def find_case_where(df_ref: pd.DataFrame, df_cmp: pd.DataFrame, *, success_ref: bool, success_cmp: bool) -> Optional[pd.Series]:
    joined = merge_runs(df_ref, df_cmp, suffixes=("_ref", "_cmp"))
    mask = (joined["is_correct_ref"] == success_ref) & (joined["is_correct_cmp"] == success_cmp)
    if not mask.any():
        return None
    return joined[mask].iloc[0]


__all__ = [
    "RunSpec",
    "load_dataset_frame",
    "load_run_dataframe",
    "discover_default_specs",
    "compute_accuracy",
    "accuracy_breakdown",
    "retrieval_stats",
    "merge_runs",
    "find_case_where",
]
