import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from RagUtils import hybrid_retrieve_documents_for_batch, prepare_generator_inputs, retrieve_documents_for_batch
import evaluate
from typing import List, Dict, Any, Tuple

# Import a type hint for models, tokenizer, and DenseRetriever
from transformers import PreTrainedModel, PreTrainedTokenizerFast

# calculate_metrics function (as defined before)
def calculate_metrics(predictions, references):
    predictions_str = [str(p) if p is not None else "" for p in predictions]
    references_str = [str(r) if r is not None else "" for r in references]

    squad_metric = evaluate.load("squad") # Loads SQuAD v1.1 metric by default
    
    formatted_predictions = [{"prediction_text": pred, "id": str(i)} for i, pred in enumerate(predictions_str)]
    # SQuAD metric expects list of lists for reference texts, even if only one ref per pred.
    formatted_references = [
        {"answers": {"text": [ref] if ref else [""], "answer_start": [-1]}, "id": str(i)}
        for i, ref in enumerate(references_str)
    ]
    try:
        if not formatted_predictions or not formatted_references:
            print("Warning: Predictions or references list is empty for metric calculation in custom_rag_eval.")
            return {"exact_match": 0.0, "f1": 0.0}
        results = squad_metric.compute(predictions=formatted_predictions, references=formatted_references)
        return {"exact_match": results["exact_match"], "f1": results["f1"]}
    except Exception as e:
        print(f"Error in metric calculation (custom_rag_eval): {e}")
        import traceback
        traceback.print_exc()
        return {"exact_match": 0.0, "f1": 0.0}


def evaluate_custom_rag_model(
    question_encoder_model: PreTrainedModel,
    dense_retriever, # Forward reference if DenseRetriever is not imported here
    generator_model: PreTrainedModel,
    eval_dataloader: torch.utils.data.DataLoader, # Corrected type hint
    question_tokenizer: PreTrainedTokenizerFast,
    generator_tokenizer: PreTrainedTokenizerFast,
    k_retrieved: int,
    max_combined_length: int,
    max_answer_length: int, # Added for generator.generate
    device: torch.device,
    epoch_num_for_log="eval",
    max_logged_examples: int = 3, # How many examples to log with retrieved docs
    K_DENSE_RETRIEVAL: int = 100, # Number of dense retrievals to fetch
    wandb_run_obj = None # Pass the wandb run object for logging
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:

    print(f"\nRunning custom RAG evaluation for {epoch_num_for_log}...")
    question_encoder_model.eval()
    generator_model.eval()

    all_final_predictions = []
    all_reference_answers = []
    logged_examples_count = 0
    logged_qa_retrieval_samples_eval = []

    if not eval_dataloader or len(eval_dataloader) == 0:
        print("Evaluation dataloader is empty. Skipping evaluation.")
        return {}, []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch_num_for_log}")):
            q_input_ids = batch["input_ids"].to(device)
            q_attention_mask = batch["attention_mask"].to(device)
            original_question_strings = batch["original_question"] # Expects list of strings
            current_reference_answers = batch["original_answer"]   # Expects list of strings
            batch_precomputed_sparse_for_eval = batch["precomputed_sparse_docs"]

            # 1. Get Query Embeddings
            query_embeddings_tuple = question_encoder_model(input_ids=q_input_ids, attention_mask=q_attention_mask)
            current_query_embeddings = query_embeddings_tuple[0]

            # 2. Retrieve Documents 
            # retrieved_info = retrieve_documents_for_batch(
            #     query_embeddings_batch=current_query_embeddings,
            #     dense_retriever=dense_retriever,
            #     k=k_retrieved,
            #     normalize_query_for_faiss=True # Assuming E5
            # )
            hybrid_retrieved_info_eval = hybrid_retrieve_documents_for_batch(
                query_embeddings_batch=current_query_embeddings,
                batch_precomputed_sparse_docs=batch_precomputed_sparse_for_eval, # <<< PASS
                dense_retriever=dense_retriever,
                final_k=k_retrieved, 
                k_dense_to_fetch=K_DENSE_RETRIEVAL,
                device=device
            )
            batch_retrieved_doc_texts = hybrid_retrieved_info_eval["retrieved_doc_texts"]
            batch_retrieved_doc_titles = hybrid_retrieved_info_eval["retrieved_doc_titles"]
            batch_retrieved_doc_embeddings = hybrid_retrieved_info_eval["retrieved_doc_embeddings"].to(device)

            # 3. Prepare Generator Inputs
            generator_inputs = prepare_generator_inputs(
                original_question_strings=original_question_strings,
                retrieved_doc_titles=batch_retrieved_doc_titles,
                retrieved_doc_texts=batch_retrieved_doc_texts,
                generator_tokenizer=generator_tokenizer,
                max_combined_length=max_combined_length,
                device=device
            )
            batch_generator_input_ids = generator_inputs["generator_input_ids"]
            batch_generator_attention_mask = generator_inputs["generator_attention_mask"]

            # 4. Generate an answer candidate for each (question, retrieved_doc) context
            candidate_generated_ids = generator_model.generate(
                input_ids=batch_generator_input_ids,
                attention_mask=batch_generator_attention_mask,
                num_beams=4, max_length=max_answer_length + 20, early_stopping=True,
                pad_token_id=generator_tokenizer.eos_token_id if generator_tokenizer.eos_token_id is not None else generator_tokenizer.pad_token_id,
                eos_token_id=generator_tokenizer.eos_token_id,
                decoder_start_token_id=generator_model.config.decoder_start_token_id
            )
            candidate_generated_ids = candidate_generated_ids.view(
                len(original_question_strings), k_retrieved, -1
            )

            # 5. Select the best answer for each original question using doc scores
            expanded_query_embeddings = current_query_embeddings.unsqueeze(1)
            doc_scores = torch.bmm(expanded_query_embeddings, batch_retrieved_doc_embeddings.transpose(1, 2)).squeeze(1)
            best_doc_indices = torch.argmax(doc_scores, dim=1)

            final_predictions_for_batch = []
            for i in range(len(original_question_strings)):
                best_doc_idx = best_doc_indices[i].item()
                selected_generated_ids = candidate_generated_ids[i, best_doc_idx, :]
                decoded_text = generator_tokenizer.decode(selected_generated_ids, skip_special_tokens=True)
                final_predictions_for_batch.append(decoded_text)
            
            all_final_predictions.extend(final_predictions_for_batch)
            all_reference_answers.extend(current_reference_answers)

            # Log a few examples with retrieved docs for wandb
            if wandb_run_obj and logged_examples_count < max_logged_examples:
                for i in range(len(original_question_strings)):
                    if logged_examples_count < max_logged_examples:
                        best_doc_idx_for_log = best_doc_indices[i].item()
                        titles_to_log = batch_retrieved_doc_titles[i]
                        texts_to_log = batch_retrieved_doc_texts[i]
                        
                        sample = {
                            "epoch": epoch_num_for_log,
                            "question": original_question_strings[i],
                            "generated_answer": final_predictions_for_batch[i],
                            "reference_answer": current_reference_answers[i],
                            "top_retrieved_doc_title": titles_to_log[best_doc_idx_for_log] if titles_to_log and best_doc_idx_for_log < len(titles_to_log) else "N/A",
                            "top_retrieved_doc_text_snippet": (texts_to_log[best_doc_idx_for_log][:150] + "...") if texts_to_log and best_doc_idx_for_log < len(texts_to_log) else "N/A",
                            "all_retrieved_titles": " | ".join(filter(None, titles_to_log))
                        }
                        logged_qa_retrieval_samples_eval.append(sample)
                        logged_examples_count += 1
                    else: break
        
    metrics = {}
    if all_reference_answers and all_final_predictions:
        if len(all_final_predictions) == len(all_reference_answers):
            metrics = calculate_metrics(all_final_predictions, all_reference_answers)
            print(f"Custom RAG Evaluation Metrics for {epoch_num_for_log}: EM = {metrics.get('exact_match',0.0):.4f}, F1 = {metrics.get('f1',0.0):.4f}")
        else:
            print(f"Warning: Mismatch in length for metric calculation. Preds: {len(all_final_predictions)}, Refs: {len(all_reference_answers)}")
    else:
        print("Not enough data for metrics in custom RAG evaluation.")

    question_encoder_model.train()
    generator_model.train()
    return metrics, logged_qa_retrieval_samples_eval