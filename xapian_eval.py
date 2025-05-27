import os

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import wandb
import numpy as np 
from tqdm.auto import tqdm
import pandas as pd 
import evaluate 
from typing import List, Dict, Any, Tuple
import json

from NqDataset import NQDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from utils import load_local_nq_json, custom_collate_fn, load_precomputed_sparse_results
from RagUtils import prepare_generator_inputs

from RagEval import calculate_metrics


def evaluate_sparse_rag_pipeline(
    generator_model: PreTrainedModel,
    eval_dataloader: torch.utils.data.DataLoader,
    generator_tokenizer: PreTrainedTokenizerFast,
    k_to_feed_generator: int, 
    max_combined_length: int,
    max_answer_length: int,  
    device: torch.device,
    epoch_num_for_log="eval_sparse_rag",
    max_logged_examples: int = 3,
    wandb_run_obj = None,

) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:

    print(f"\nRunning Sparse-Only RAG Pipeline Evaluation for {epoch_num_for_log}...")
    generator_model.eval()

    all_final_predictions = []
    all_reference_answers = []
    
    logged_examples_count = 0
    logged_qa_retrieval_samples_eval = []

    if not eval_dataloader or len(eval_dataloader) == 0:
        print("Evaluation dataloader is empty. Skipping evaluation.")
        return {}, []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch_num_for_log} (Sparse RAG)")):
            original_question_strings = batch["original_question"] # List of strings
            current_reference_answers = batch["original_answer"]   # List of strings
            batch_precomputed_sparse_docs = batch["precomputed_sparse_docs"] # List of lists of dicts

            batch_size = len(original_question_strings)
            
            # Select top k_to_feed_generator documents based on sparse_score for each question
            selected_sparse_texts_for_batch = [[] for _ in range(batch_size)]
            selected_sparse_titles_for_batch = [[] for _ in range(batch_size)]
            top_sparse_scores_for_batch = [[] for _ in range(batch_size)] # For answer selection & logging

            for i in range(batch_size):
                # Sort precomputed sparse docs by 'sparse_score' (higher is better)
                # The precomputed list might already be sorted. If not, sort here.
                # Assuming sparse_results from Xapian were already sorted by score.
                
                current_item_sparse_docs = sorted(
                    batch_precomputed_sparse_docs[i], 
                    key=lambda x: x.get("sparse_score", 0.0), 
                    reverse=True
                )

                for j in range(k_to_feed_generator):
                    if j < len(current_item_sparse_docs):
                        doc_info = current_item_sparse_docs[j]
                        selected_sparse_texts_for_batch[i].append(doc_info.get("text", ""))
                        selected_sparse_titles_for_batch[i].append(doc_info.get("title", ""))
                        top_sparse_scores_for_batch[i].append(doc_info.get("sparse_score", 0.0))
                    else: # If fewer than k_to_feed_generator docs were pre-retrieved
                        selected_sparse_texts_for_batch[i].append("")
                        selected_sparse_titles_for_batch[i].append("")
                        top_sparse_scores_for_batch[i].append(0.0) # Or a very low score

            # 3. Prepare Generator Inputs using these selected sparse docs
            generator_inputs = prepare_generator_inputs(
                original_question_strings, 
                selected_sparse_titles_for_batch,
                selected_sparse_texts_for_batch,
                generator_tokenizer, 
                max_combined_length, 
                device
            )
            batch_generator_input_ids = generator_inputs["generator_input_ids"]
            batch_generator_attention_mask = generator_inputs["generator_attention_mask"]
            # Shape: [batch_size * k_to_feed_generator, seq_len]

            # 4. Generate an answer candidate for each (question, sparse_doc) context
            candidate_generated_ids = generator_model.generate(
                input_ids=batch_generator_input_ids,
                attention_mask=batch_generator_attention_mask,
                num_beams=4, max_length=max_answer_length + 20, early_stopping=True,
                pad_token_id=generator_tokenizer.eos_token_id if generator_tokenizer.eos_token_id is not None else generator_tokenizer.pad_token_id,
                eos_token_id=generator_tokenizer.eos_token_id,
                decoder_start_token_id=generator_model.config.decoder_start_token_id
            )
            candidate_generated_ids = candidate_generated_ids.view(
                batch_size, k_to_feed_generator, -1 # [batch_size, k_to_feed_generator, seq_len]
            )
            
            all_candidate_texts_for_batch = []
            for i in range(batch_size):
                candidates_for_item = [
                    generator_tokenizer.decode(candidate_generated_ids[i, j, :], skip_special_tokens=True)
                    for j in range(k_to_feed_generator)
                ]
                all_candidate_texts_for_batch.append(candidates_for_item)

            # 5. Select the best answer using the sparse_scores of the contexts
            final_predictions_for_batch = []
            for i in range(batch_size):
                if top_sparse_scores_for_batch[i]:
                    best_context_idx = np.argmax(top_sparse_scores_for_batch[i]) # Highest sparse score
                    final_predictions_for_batch.append(all_candidate_texts_for_batch[i][best_context_idx])
                elif all_candidate_texts_for_batch[i]: # Fallback: if no scores but candidates exist, take first
                    final_predictions_for_batch.append(all_candidate_texts_for_batch[i][0])
                else: # No candidates generated
                    final_predictions_for_batch.append("") 
            
            all_final_predictions.extend(final_predictions_for_batch)
            all_reference_answers.extend(current_reference_answers)

            # Log a few examples with retrieved docs for wandb
            if wandb_run_obj and logged_examples_count < max_logged_examples and batch_idx == 0:
                for i in range(batch_size):
                    if logged_examples_count < max_logged_examples:
                        best_context_idx_for_log = np.argmax(top_sparse_scores_for_batch[i]) if top_sparse_scores_for_batch[i] else 0
                        
                        sample_log = {
                            "epoch": epoch_num_for_log,
                            "question": original_question_strings[i],
                            "generated_answer": final_predictions_for_batch[i],
                            "reference_answer": current_reference_answers[i] if i < len(current_reference_answers) else "N/A",
                        }
                        # Add details for all k_to_feed_generator sparse documents
                        for doc_rank in range(k_to_feed_generator):
                            title = selected_sparse_titles_for_batch[i][doc_rank]
                            text = selected_sparse_texts_for_batch[i][doc_rank]
                            score = top_sparse_scores_for_batch[i][doc_rank]
                            
                            sample_log[f"doc_{doc_rank+1}_title"] = title
                            sample_log[f"doc_{doc_rank+1}_text_snippet"] = (text[:100] + "...") if text else "N/A"
                            sample_log[f"doc_{doc_rank+1}_sparse_score"] = round(score, 4)
                        
                        logged_qa_retrieval_samples_eval.append(sample_log)
                        logged_examples_count += 1
                    else: break
        
    metrics = {}
    if all_reference_answers and all_final_predictions:
        if len(all_final_predictions) == len(all_reference_answers):
            metrics = calculate_metrics(all_final_predictions, all_reference_answers)
            print(f"Sparse-Only RAG Evaluation Metrics for {epoch_num_for_log}: EM = {metrics.get('exact_match',0.0):.4f}, F1 = {metrics.get('f1',0.0):.4f}")
        else: print(f"Warning: Mismatch in length for metric calc. Preds: {len(all_final_predictions)}, Refs: {len(all_reference_answers)}")
    else: print("Not enough data for metrics in sparse-only RAG evaluation.")

    generator_model.train()
    return metrics, logged_qa_retrieval_samples_eval

def run_sparse_rag_evaluation_test():
    
    global current_wandb_run

    # Configuration for the evaluation
    CONFIG = {
        "nq_dev_file": "downloads/data/gold_passages_info/nq_dev.json",
        "offline_sparse_results_file": "downloads/data/nq_dev_sparse_retrieval.jsonl",
        "max_eval_samples": None,
        "k_to_feed_generator": 50,
        "max_combined_length_gen": 512, 
        "max_answer_length_gen": 128,  
        "max_question_length_for_dataset": 128,
        "max_answer_length_for_dataset": 64,
    }
    generator_path = "best_bart_model" 
    encoder_path = "models/retriever_finetuned_e5_best"
    EVAL_BATCH_SIZE = 4
    MAX_QUESTION_LENGTH = CONFIG["max_question_length_for_dataset"]
    MAX_ANSWER_LENGTH = CONFIG["max_answer_length_for_dataset"]

    print("--- Starting Standalone Xapian Evaluation Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using device: {device}")
    
    # 1. Load BART Generator and Tokenizer
    print(f"Loading BART generator tokenizer from: {generator_path}")
    bart_tokenizer = AutoTokenizer.from_pretrained(generator_path)
    print(f"Loading BART Generator: {generator_path}")
    generator = AutoModelForSeq2SeqLM.from_pretrained(generator_path).to(device)
    # Set BOS token if needed for BART
    if hasattr(generator.config, "forced_bos_token_id") and \
       generator.config.forced_bos_token_id is None and \
       generator.config.bos_token_id is not None:
        generator.config.forced_bos_token_id = generator.config.bos_token_id
    print("BART Generator and Tokenizer loaded.")

    # 2. Load NQ Evaluation Data and Pre-computed Sparse Results
    print("Loading NQ evaluation data...")
    eval_data_list = load_local_nq_json(CONFIG["nq_dev_file"], limit=CONFIG["max_eval_samples"])
    
    print("Loading pre-computed sparse retrieval data...")
    eval_sparse_data_lookup = load_precomputed_sparse_results(CONFIG["offline_sparse_results_file"]) 

    if not eval_data_list: print("Evaluation data list is empty."); return
        
    dummy_q_tokenizer_for_dataset = AutoTokenizer.from_pretrained(encoder_path)

    eval_dataset = NQDataset(
        eval_data_list, 
        eval_sparse_data_lookup, # Pass the loaded sparse data
        question_tokenizer=dummy_q_tokenizer_for_dataset, # Or your actual E5 tokenizer
        generator_tokenizer=bart_tokenizer, 
        max_question_length=MAX_QUESTION_LENGTH, # From CONFIG
        max_answer_length=MAX_ANSWER_LENGTH    # From CONFIG
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, collate_fn=custom_collate_fn)
    if len(eval_dataloader) == 0: print("Evaluation DataLoader is empty."); return
    print("Evaluation DataLoader created.")

    # (Optional) Initialize WandB
    current_wandb_run = wandb.init(
        project="RAG_Evaluation",
        name="Sparse_RAG_Eval_Test",
        config=CONFIG,
        reinit=True,
        mode="offline" if not wandb.run else "online" 
    )

    # 3. Call the sparse-only RAG evaluation function
    evaluate_sparse_rag_pipeline(
        generator_model=generator,
        eval_dataloader=eval_dataloader,
        generator_tokenizer=bart_tokenizer,
        k_to_feed_generator=CONFIG["k_to_feed_generator"],
        max_combined_length=CONFIG["max_combined_length_gen"],
        max_answer_length=CONFIG["max_answer_length_gen"],
        device=device,
        epoch_num_for_log="sparse_only_test",
        max_logged_examples=100,
        wandb_run_obj=current_wandb_run 
    )

    if current_wandb_run: current_wandb_run.finish()

if __name__ == "__main__":

    def load_precomputed_sparse_results(jsonl_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        print(f"Mock loading sparse results from {jsonl_file_path}")

        loaded_data = {}
        if os.path.exists(jsonl_file_path):
            with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        key = entry.get("example_id", entry.get("original_question"))
                        if key:
                            loaded_data[key] = entry.get("sparse_retrieved_docs", [])
                    except json.JSONDecodeError:
                        continue # Skip malformed lines
        else:
            print(f"Warning: Sparse results file not found at {jsonl_file_path}, returning empty lookup.")
        return loaded_data

    run_sparse_rag_evaluation_test()
