import json
import re
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)  # remove articles
    s = ''.join(ch for ch in s if ch not in string.punctuation)  # remove punctuations
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_bleu(prediction, references):
    smoothie = SmoothingFunction().method4
    ref_list = [normalize_answer(ref).split() for ref in references]
    pred_tokens = normalize_answer(prediction).split()
    return sentence_bleu(ref_list, pred_tokens, smoothing_function=smoothie)

def compute_rouge_l(prediction, references):
    rouge = Rouge()
    try:
        scores = [rouge.get_scores(prediction, ref)[0]["rouge-l"]["f"] for ref in references]
        return max(scores)  # Take best matching reference
    except ValueError:
        return 0.0


def max_metric_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)

def evaluate_pipeline(pipeline, test_set, verbose=False, log_path="predictions_log.json", strategy="thorough", top_k=None):
    em_scores = []
    f1_scores = []
    bleu_scores = []
    rouge_l_scores = []
    logs = []

    for i, sample in enumerate(test_set):
        query = sample["query"]
        references = sample["answers"]

        prediction = pipeline.generate_answer(query, strategy=strategy, top_k=top_k)

        em = max_metric_over_ground_truths(exact_match, prediction, references)
        f1 = max_metric_over_ground_truths(compute_f1, prediction, references)
        bleu = compute_bleu(prediction, references)
        rouge_l = compute_rouge_l(prediction, references)

        em_scores.append(em)
        f1_scores.append(f1)
        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l)

        logs.append({
            "query": query,
            "references": references,
            "prediction": prediction,
            "EM": em,
            "F1": round(f1, 3),
            "BLEU": round(bleu, 3),
            "ROUGE-L": round(rouge_l, 3)
        })

        if verbose:
            print(f"[{i}] Query: {query}")
            print(f"→ Prediction: {prediction}")
            print(f"✓ References: {references}")
            print(f"  EM: {em}, F1: {round(f1, 3)}, BLEU: {round(bleu, 3)}, ROUGE-L: {round(rouge_l, 3)}\n")

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return {
        "EM": round(avg_em * 100, 2),
        "F1": round(avg_f1 * 100, 2),
        "BLEU": round(avg_bleu * 100, 2),
        "ROUGE-L": round(avg_rouge_l * 100, 2)
    }
