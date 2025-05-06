import json
import re
import string
from nltk.stem import WordNetLemmatizer
from collections import Counter
import wandb

import nltk
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\s+', ' ', s).strip()
    return ' '.join([lemmatizer.lemmatize(word) for word in s.split()])

def exact_match(prediction, ground_truth):
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)

    # Only return True if both are non-empty and equal, or both truly empty
    if not norm_pred and not norm_gold:
        return prediction.strip() == ground_truth.strip()  # strict empty match
    return norm_pred == norm_gold


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_metric_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)

def evaluate_pipeline(pipeline, test_set, verbose=False, log_path="predictions_log.json", strategy="thorough", top_k=None):
    em_scores = []
    f1_scores = []
    logs = []

    for i, sample in enumerate(test_set):
        query = sample["query"]
        references = sample["answers"]

        prediction = pipeline.generate_answer(query, strategy=strategy, top_k=top_k)

        wandb.log({
            "sample_query": query,
            "sample_reference": references,
            "sample_prediction": prediction,
            "stage": "Evaluation",
            "step": i
        })

        em = max_metric_over_ground_truths(exact_match, prediction, references)
        f1 = max_metric_over_ground_truths(compute_f1, prediction, references)

       

        em_scores.append(em)
        f1_scores.append(f1)


        logs.append({
            "query": query,
            "references": references,
            "prediction": prediction,
            "EM": em,
            "F1": round(f1, 3),
        })

        if verbose:
            print(f"[{i}] Query: {query}")
            print(f"→ Prediction: {prediction}")
            print(f"✓ References: {references}")
            print(f"  EM: {em}, F1: {round(f1, 3)} \n")

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    return {
        "EM": round(avg_em * 100, 2),
        "F1": round(avg_f1 * 100, 2)
    }
