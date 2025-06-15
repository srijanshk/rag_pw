import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json
import re
import torch
import wandb
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import SequenceMatcher
from tqdm import tqdm

# ——————————————————————————————————————————————————————————————————————————————— #
# 1. Configuration (EDIT THESE PATHS AS NEEDED)
# ——————————————————————————————————————————————————————————————————————————————— #

# Local dataset paths
GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
MATH_BASE_DIR    = "./thesis_datasets/math_hendrycks"
MATH_SUBJECTS    = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Model name (Hugging Face repository) for LLaMA-3.1-8b-Instruct
MODEL_NAME = "meta-llama/Llama-3.1-8b-Instruct"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation settings: zero-shot CoT, greedy decoding
GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,      # Greedy decoding
    "pad_token_id": 0,       # Assumes tokenizer.pad_token_id == 0
}


# ——————————————————————————————————————————————————————————————————————————————— #
# 2. Utility Functions for Parsing & Metrics
# ——————————————————————————————————————————————————————————————————————————————— #

def extract_final_numeric_answer(generated_text: str) -> str:
    """
    Heuristic: Find the last integer or decimal number in the generated text.
    Returns that number as a string, or "" if none found.
    """
    tokens = re.findall(r"-?\d+\.\d+|-?\d+", generated_text)
    return tokens[-1] if tokens else ""


def split_reasoning_and_answer(generated_text: str) -> (str, str):
    """
    Wrap the entire generated chain-of-thought in <reasoning> tags, and
    put the extracted numeric answer inside <answer> tags.
    Returns a tuple (reasoning_str, answer_str).
    """
    answer_token = extract_final_numeric_answer(generated_text)
    reasoning_block = generated_text.strip()
    answer_block = answer_token.strip()
    return reasoning_block, answer_block


def extract_gsm8k_gold_answer(answer_field: str) -> str:
    """
    From GSM8K 'answer' field (which contains computations and ends with '#### <num>'),
    extract the gold numeric answer after '#### '. If not found, fall back to last numeric token.
    """
    m = re.search(r"####\s*([-\d\.]+)", answer_field)
    if m:
        return m.group(1)
    tokens = re.findall(r"-?\d+\.\d+|-?\d+", answer_field)
    return tokens[-1] if tokens else ""


def extract_math_gold_answer(solution_field: str) -> str:
    """
    From MATH 'solution' field (which may contain '\\boxed{<num>}' or plain text),
    extract the gold numeric answer. First try '\\boxed{num}', then fallback to last numeric token.
    """
    m = re.search(r"\\boxed\{([-\d\.]+)\}", solution_field)
    if m:
        return m.group(1)
    tokens = re.findall(r"-?\d+\.\d+|-?\d+", solution_field)
    return tokens[-1] if tokens else ""


def compute_answer_accuracy(preds: list, golds: list) -> float:
    """
    Exact-match accuracy between predicted answers (strings) and gold answers (strings).
    """
    assert len(preds) == len(golds), "Predictions and gold lists must be same length."
    correct = sum(1 for p, g in zip(preds, golds) if p.strip() == g.strip())
    return correct / len(preds) if preds else 0.0


def compute_reasoning_similarity(pred_rationales: list, gold_rationales: list) -> float:
    """
    Compute average SequenceMatcher ratio between predicted and gold rationales.
    Returns None if gold_rationales are all empty.
    """
    assert len(pred_rationales) == len(gold_rationales)
    total_ratio = 0.0
    count = 0
    for pr, gr in zip(pred_rationales, gold_rationales):
        if not pr.strip() and not gr.strip():
            ratio = 1.0
        else:
            ratio = SequenceMatcher(None, pr, gr).ratio()
        total_ratio += ratio
        count += 1
    return (total_ratio / count) if count > 0 else None


def compute_arithmetic_step_accuracy(pred_rationales: list) -> float:
    """
    For each rationale, find all arithmetic expressions of form:
        <int> <op> <int> = <int>
    where op ∈ {+,-,*,/}. Check if RHS matches the actual computation.
    Returns (correct_steps / total_steps) or 0.0 if no steps found.
    """
    step_pattern = re.compile(r"(-?\d+)\s*([\+\-\*/])\s*(-?\d+)\s*=\s*(-?\d+)")
    total_steps = 0
    correct_steps = 0

    for rationale in pred_rationales:
        for match in step_pattern.finditer(rationale):
            a_str, op, b_str, res_str = match.groups()
            a, b, res = int(a_str), int(b_str), int(res_str)
            total_steps += 1

            computed = None
            if op == "+":
                computed = a + b
            elif op == "-":
                computed = a - b
            elif op == "*":
                computed = a * b
            elif op == "/":
                if b != 0 and a % b == 0:
                    computed = a // b

            if computed == res:
                correct_steps += 1

    return (correct_steps / total_steps) if total_steps > 0 else 0.0


# ——————————————————————————————————————————————————————————————————————————————— #
# 3. Functions to Load Local Datasets
# ——————————————————————————————————————————————————————————————————————————————— #

def load_gsm8k_local(path: str):
    """
    Load GSM8K test examples from a JSONL file at `path`.
    Expects each line to be a JSON object with keys:
      - "question"
      - "answer" (string containing computations and '#### <num>')
      - optionally "original_solution", "solution", or "explanation"

    Returns: (questions, answers, rationales)
    """
    questions = []
    answers = []
    rationales = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("question", "").strip()
            raw_ans = obj.get("answer", "").strip()
            gold_ans = extract_gsm8k_gold_answer(raw_ans)
            # For rationale, try "original_solution", else "solution"/"explanation"
            r = obj.get("original_solution", "") or obj.get("solution", "") or obj.get("explanation", "")
            questions.append(q)
            answers.append(gold_ans)
            rationales.append(r.strip())

    return questions, answers, rationales


def load_math_hendrycks_local(base_dir: str, subjects: list):
    """
    Load Hendrycks Math test examples from local directories.
    For each subject in `subjects`, expects: base_dir/subject/test.jsonl
    Each JSON line should have:
      - "problem"
      - "solution" (string containing explanations and '\\boxed{num}' or numeric answer)

    Returns: (questions, answers, rationales)
    """
    all_questions = []
    all_answers = []
    all_rationales = []

    for subject in subjects:
        subj_dir = os.path.join(base_dir, subject)
        test_file = os.path.join(subj_dir, "test.jsonl")

        if not os.path.isfile(test_file):
            print(f"Warning: expected file not found: {test_file}. Skipping subject '{subject}'.")
            continue

        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                q = obj.get("problem", "").strip()
                sol = obj.get("solution", "").strip()
                gold_ans = extract_math_gold_answer(sol)
                all_questions.append(q)
                all_answers.append(gold_ans)
                all_rationales.append(sol)

    return all_questions, all_answers, all_rationales


# ——————————————————————————————————————————————————————————————————————————————— #
# 4. Inference & Evaluation on Lists of Examples
# ——————————————————————————————————————————————————————————————————————————————— #

def evaluate_on_lists(model, tokenizer, questions: list, answers: list, rationales: list,
                      prompt_prefix: str = "", dataset_name: str = "") -> dict:
    """
    Run zero-shot chain-of-thought inference on lists of (questions, answers, rationales).
    Builds a list of per-example dicts with fields:
      - question
      - gold_answer
      - generated_reasoning
      - generated_answer
      - formatted_output (with <reasoning>...</reasoning><answer>...</answer>)

    Returns:
      {
        "examples": [ { ... per-example dict ... }, ... ],
        "metrics": {
          "accuracy": float,
          "reasoning_similarity": float or None,
          "arithmetic_accuracy": float,
          "n_examples": int
        }
      }
    """
    n_examples = len(questions)
    preds = []
    pred_rationales = []
    per_example_data = []

    print(f"Running inference on {n_examples} examples from {dataset_name}...")

    # Use tqdm to visualize progress
    for idx, (q, gold_ans, gold_rat) in enumerate(
            tqdm(zip(questions, answers, rationales),
                 total=n_examples,
                 desc=f"Inference {dataset_name}")):

        prompt = prompt_prefix + "Q: " + q + "\n" + "Let's think step by step.\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **GENERATION_KWARGS)

        # Decode only newly generated tokens (skip prompt tokens)
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        # Split into reasoning and answer
        reasoning_block, answer_block = split_reasoning_and_answer(generated)

        # Store predicted answer for accuracy
        preds.append(answer_block)
        pred_rationales.append(reasoning_block)

        # Build formatted output
        formatted_output = (
            "<reasoning>\n"
            + reasoning_block
            + "\n</reasoning>\n"
            + "<answer>\n"
            + answer_block
            + "\n</answer>"
        )

        per_example_data.append({
            "question": q,
            "gold_answer": gold_ans,
            "generated_reasoning": reasoning_block,
            "generated_answer": answer_block,
            "formatted_output": formatted_output
        })

    # Compute metrics
    final_acc = compute_answer_accuracy(preds, answers)
    reasoning_sim = compute_reasoning_similarity(pred_rationales, rationales) if any(rationales) else None
    arithmetic_acc = compute_arithmetic_step_accuracy(pred_rationales)

    metrics = {
        "accuracy": final_acc,
        "reasoning_similarity": reasoning_sim,
        "arithmetic_accuracy": arithmetic_acc,
        "n_examples": n_examples,
    }

    return {
        "examples": per_example_data,
        "metrics": metrics
    }


# ——————————————————————————————————————————————————————————————————————————————— #
# 5. Main: Load Model, Load Data, Evaluate, Log & Save
# ——————————————————————————————————————————————————————————————————————————————— #

def main():
    # 5.1) Initialize Weights & Biases
    wandb.init(
        project="llama_baseline_evaluation",
        name="llama3.1-8b_instruct_gsm8k_math",
        config={
            "model_name": MODEL_NAME,
            "datasets": ["GSM8K", "Math_Hendrycks"],
            "evaluation_type": "zero-shot_CoT_greedy"
        }
    )

    # 5.2) Load tokenizer & model
    print("Loading LLaMA-3.1-8b-Instruct model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    # 5.3) Load GSM8K locally
    print(f"\nLoading GSM8K test set from: {GSM8K_TEST_FILE}")
    gsm_questions, gsm_answers, gsm_rationales = load_gsm8k_local(GSM8K_TEST_FILE)
    print(f"  • Loaded {len(gsm_questions)} GSM8K examples.\n")

    # 5.4) Load Hendrycks Math locally
    print(f"Loading Hendrycks Math test sets from: {MATH_BASE_DIR}")
    math_questions, math_answers, math_rationales = load_math_hendrycks_local(
        base_dir=MATH_BASE_DIR,
        subjects=MATH_SUBJECTS
    )
    print(f"  • Loaded {len(math_questions)} Math examples across {len(MATH_SUBJECTS)} subjects.\n")

    # 5.5) Evaluate on GSM8K
    print("=== Evaluating on GSM8K ===")
    gsm_result = evaluate_on_lists(
        model=model,
        tokenizer=tokenizer,
        questions=gsm_questions,
        answers=gsm_answers,
        rationales=gsm_rationales,
        prompt_prefix="",
        dataset_name="GSM8K"
    )
    gsm_examples = gsm_result["examples"]
    gsm_metrics = gsm_result["metrics"]

    # 5.6) Evaluate on Hendrycks Math
    print("\n=== Evaluating on Hendrycks Math (All Subjects) ===")
    math_result = evaluate_on_lists(
        model=model,
        tokenizer=tokenizer,
        questions=math_questions,
        answers=math_answers,
        rationales=math_rationales,
        prompt_prefix="",
        dataset_name="Math_Hendrycks"
    )
    math_examples = math_result["examples"]
    math_metrics = math_result["metrics"]

    # 5.7) Log Metrics to wandb
    wandb.log({
        "GSM8K/accuracy": gsm_metrics["accuracy"],
        "GSM8K/reasoning_similarity": gsm_metrics["reasoning_similarity"] or 0.0,
        "GSM8K/arithmetic_accuracy": gsm_metrics["arithmetic_accuracy"],
        "Math/accuracy": math_metrics["accuracy"],
        "Math/reasoning_similarity": math_metrics["reasoning_similarity"] or 0.0,
        "Math/arithmetic_accuracy": math_metrics["arithmetic_accuracy"],
    })

    # 5.8) Create DataFrames for per-example results
    gsm_df = pd.DataFrame.from_records(gsm_examples)
    math_df = pd.DataFrame.from_records(math_examples)

    # Add a column to identify dataset
    gsm_df["dataset"] = "GSM8K"
    math_df["dataset"] = "Math_Hendrycks"

    combined_df = pd.concat([gsm_df, math_df], ignore_index=True)

    # 5.9) Save combined results to CSV locally
    output_csv = "llama_baseline_outputs.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"\nSaved per-example results to: {output_csv}")

    # 5.10) Log the table to wandb
    table = wandb.Table(dataframe=combined_df)
    wandb.log({"results_table": table})

    # 5.11) Print Summary
    print("\n" + "=" * 60)
    print("Baseline Evaluation Results on LLaMA-3.1-8b-Instruct")
    print("=" * 60 + "\n")

    # GSM8K Summary
    print(f"GSM8K ({gsm_metrics['n_examples']} examples):")
    print(f"  → Final Answer Accuracy     : {gsm_metrics['accuracy'] * 100:.2f}%")
    if gsm_metrics["reasoning_similarity"] is not None:
        print(f"  → Reasoning Similarity (≈)  : {gsm_metrics['reasoning_similarity'] * 100:.2f}%")
    else:
        print("  → Reasoning Similarity (≈)  : N/A (no gold rationales provided)")
    print(f"  → Arithmetic Step Accuracy  : {gsm_metrics['arithmetic_accuracy'] * 100:.2f}%")
    print("-" * 60)

    # Math Summary
    print(f"Hendrycks Math ({math_metrics['n_examples']} examples):")
    print(f"  → Final Answer Accuracy     : {math_metrics['accuracy'] * 100:.2f}%")
    if math_metrics["reasoning_similarity"] is not None:
        print(f"  → Reasoning Similarity (≈)  : {math_metrics['reasoning_similarity'] * 100:.2f}%")
    else:
        print("  → Reasoning Similarity (≈)  : N/A (no gold rationales provided)")
    print(f"  → Arithmetic Step Accuracy  : {math_metrics['arithmetic_accuracy'] * 100:.2f}%")
    print("=" * 60 + "\n")

    # 5.12) Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()

