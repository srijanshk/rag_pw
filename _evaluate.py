import json
import re
import string
from nltk.stem import WordNetLemmatizer
from collections import Counter
import wandb
import logging # Added for logging
from tqdm import tqdm

import nltk
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
logger_eval = logging.getLogger(__name__)
if not logger_eval.hasHandlers():
    logging.basicConfig(level=logging.INFO)


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

def evaluate_pipeline(pipeline, test_set, verbose=False, log_path="predictions_log.json", strategy=None, top_k=None, epoch=None, batch_size: int = 16, on_batch = False):
    em_scores = []
    f1_scores = []
    logs = []

    if wandb.run:
        table = wandb.Table(columns=["query", "references", "prediction", "Epoch" ])

    if on_batch:
        # 1) Split test_set into chunks of size batch_size
        for start in tqdm(range(0, len(test_set), batch_size), desc="Batches"):
            chunk = test_set[start : start + batch_size]
            queries    = [sample["query"]   for sample in chunk]
            references = [sample["answers"] for sample in chunk]

            # 2) Bulk-generate answers for this chunk
            preds = pipeline.generate_answers_batch(
                queries,
                strategy=strategy,
                top_k=top_k,
                use_sampling=False,
                num_beams_generation=3,
                max_length_generation=128
                
            )

            # 3) Evaluate each item in the chunk
            for i, (query, refs, pred) in enumerate(zip(queries, references, preds), start=start):
                # Log to wandb table if active
                if wandb.run:
                    table.add_data(query, refs, pred, epoch)

                # Compute metrics
                em = max_metric_over_ground_truths(exact_match, pred, refs)
                f1 = max_metric_over_ground_truths(compute_f1,     pred, refs)

                em_scores.append(em)
                f1_scores.append(f1)

                logs.append({
                    "query":       query,
                    "references":  refs,
                    "prediction":  pred,
                    "EM":          em,
                    "F1":          round(f1, 3),
                })

                if verbose:
                    print(f"[{i}] Query: {query}")
                    print(f"→ Prediction: {pred}")
                    print(f"✓ References: {refs}")
                    print(f"  EM: {em}, F1: {round(f1, 3)} \n")

        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)

        avg_em = sum(em_scores) / len(em_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)

        return {
            "EM": round(avg_em * 100, 2),
            "F1": round(avg_f1 * 100, 2)
        }
    else:
        # Evaluate without batching
        for i, sample in enumerate(tqdm(test_set, desc="Evaluating Pipeline")):
            query = sample["query"]
            references = sample["answers"]
            

            prediction = pipeline.generate_answer(query, strategy=strategy, top_k=top_k)

            if wandb.run:
                table.add_data(query, references, prediction, epoch)

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

    

def evaluate_pipeline_sparse(pipeline,
                      test_set: list, # Expect a list of dicts
                      verbose=False,
                      log_path="predictions_log.json",
                      strategy="thorough", # May not be used by all pipelines
                      top_k=None,          # For generate_answer if live retrieval
                      epoch=None,          # For logging
                      use_pre_retrieved_contexts_for_eval=False, # New flag
                      pre_retrieved_contexts_key="retrieved_contexts"): # Key for pre-retrieved data in test_set items
    """
    Evaluates the RAG pipeline.
    If use_pre_retrieved_contexts_for_eval is True, it will try to find
    pre-retrieved contexts in each sample of test_set under pre_retrieved_contexts_key
    and pass them to pipeline.generate_answer().
    """
    em_scores = []
    f1_scores = []
    logs = []

    if not test_set:
        logger_eval.warning("Test set is empty. Skipping evaluation.")
        return {"EM": 0.0, "F1": 0.0}

    wandb_table = None
    if wandb.run:
        table_columns = ["query", "references", "prediction", "EM", "F1"]
        if epoch is not None:
            table_columns.append("Epoch")
        wandb_table = wandb.Table(columns=table_columns)
        wandb_log_interval = max(1, len(test_set) // 20) # Log ~20 samples or at least 1
        log_counter = 0

    logger_eval.info(f"Starting evaluation on {len(test_set)} samples. "
                     f"Using pre-retrieved contexts: {use_pre_retrieved_contexts_for_eval}")

    for i, sample in enumerate(tqdm(test_set, desc="Evaluating Pipeline")):
        query = sample.get("query")
        references = sample.get("answers", []) # Ensure it's a list

        if not query or not references:
            logger_eval.warning(f"Skipping sample {i} due to missing query or references.")
            continue

        prediction = ""
        generate_kwargs = {
            "strategy": strategy, # Pass along if pipeline uses it
            "top_k": top_k      # For live retrieval's k, or to limit pre-retrieved
        }

        if use_pre_retrieved_contexts_for_eval:
            pre_retrieved_docs = sample.get(pre_retrieved_contexts_key)
            if pre_retrieved_docs is not None: # Check if None, empty list is acceptable
                generate_kwargs["pre_retrieved_docs_with_scores"] = pre_retrieved_docs
                if verbose and i < 5: # Log for first few samples
                    logger_eval.info(f"Sample {i}: Using {len(pre_retrieved_docs)} pre-retrieved contexts for query '{query[:30]}...'")
            elif verbose and i < 5:
                 logger_eval.warning(f"Sample {i}: Flag use_pre_retrieved_contexts_for_eval is True, but "
                                     f"key '{pre_retrieved_contexts_key}' not found or is None in sample for query '{query[:30]}...'. "
                                     f"Pipeline will perform live retrieval if implemented.")


        try:
            # Pass the pre_retrieved_docs to generate_answer if the flag is set
            # The pipeline.generate_answer needs to be able to accept this kwarg
            prediction = pipeline.generate_answer(query, **generate_kwargs)
        except Exception as e:
            logger_eval.error(f"Error during generate_answer for query '{query[:50]}...': {e}", exc_info=True)
            prediction = "[ERROR DURING GENERATION]"


        em = max_metric_over_ground_truths(exact_match, prediction, references)
        f1 = max_metric_over_ground_truths(compute_f1, prediction, references)

        em_scores.append(em)
        f1_scores.append(f1)

        log_entry = {
            "query": query,
            "references": references,
            "prediction": prediction,
            "EM": em,
            "F1": round(f1, 3), # Keep F1 rounded for JSON log
        }
        if epoch is not None:
            log_entry["epoch_num"] = epoch
        logs.append(log_entry)

        if wandb_table is not None and (i % wandb_log_interval == 0 or i == len(test_set) - 1):
            wandb_row = [query, str(references), prediction, em, f1] # F1 for table can be float
            if epoch is not None:
                wandb_row.append(epoch)
            try:
                wandb_table.add_data(*wandb_row)
            except Exception as e:
                logger_eval.warning(f"Failed to add data to wandb.Table: {e}. Row: {wandb_row}")


        if verbose and i < 10: # Print for the first few samples
            print(f"[{i}] Query: {query}")
            print(f"  → Prediction: {prediction}")
            print(f"  ✓ References: {references}")
            print(f"    EM: {em*100:.2f}%, F1: {f1*100:.2f}% \n")

    # Save detailed prediction logs
    try:
        with open(log_path, "w", encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        logger_eval.info(f"Detailed prediction logs saved to: {log_path}")
    except Exception as e:
        logger_eval.error(f"Failed to save prediction logs to {log_path}: {e}", exc_info=True)


    # Log the table to wandb after the loop
    if wandb.run and wandb_table is not None:
        try:
            wandb.log({"evaluation_samples_table": wandb_table}, step=epoch if epoch is not None else None)
        except Exception as e:
            logger_eval.error(f"Failed to log wandb.Table: {e}")


    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    logger_eval.info(f"Evaluation complete. Average EM: {avg_em*100:.2f}%, Average F1: {avg_f1*100:.2f}%")

    return {
        "EM": round(avg_em * 100, 2),
        "F1": round(avg_f1 * 100, 2)
    }
