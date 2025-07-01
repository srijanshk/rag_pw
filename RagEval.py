import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from RagUtils import hybrid_retrieve_documents_for_batch, prepare_generator_inputs, retrieve_documents_for_batch
import evaluate
from typing import List, Dict, Any, Tuple
from evaluate import load as load_metric
import re

from transformers import PreTrainedModel, PreTrainedTokenizerFast

# calculate_metrics function
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
    
def evaluate_dense_rag_model(
    question_encoder_model: PreTrainedModel,
    dense_retriever, 
    generator_model: PreTrainedModel,
    eval_dataloader: torch.utils.data.DataLoader,
    question_tokenizer: PreTrainedTokenizerFast,
    generator_tokenizer: PreTrainedTokenizerFast,
    k_retrieved: int,
    max_combined_length: int,
    max_answer_length: int,
    device: torch.device,
    epoch_num_for_log="eval",
    max_logged_examples: int = 3, 
    K_DENSE_RETRIEVAL: int = 100,
    wandb_run_obj = None 
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
            original_question_strings = batch["original_question"] 
            current_reference_answers = batch["original_answer"]

            # 1. Get Query Embeddings
            query_embeddings_tuple = question_encoder_model(input_ids=q_input_ids, attention_mask=q_attention_mask)
            current_query_embeddings = query_embeddings_tuple[0]

            # 2. Retrieve Documents 
            # For Dense Only
            retrieved_info = retrieve_documents_for_batch(
                query_embeddings_batch=current_query_embeddings,
                dense_retriever=dense_retriever,
                k=k_retrieved,
                normalize_query_for_faiss=True # Assuming E5
            )
   
            batch_retrieved_doc_texts = retrieved_info["retrieved_doc_texts"]
            batch_retrieved_doc_titles = retrieved_info["retrieved_doc_titles"]
            batch_retrieved_doc_embeddings = retrieved_info["retrieved_doc_embeddings"].to(device)

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


def evaluate_custom_rag_model(
    question_encoder_model: PreTrainedModel,
    dense_retriever, 
    generator_model: PreTrainedModel,
    eval_dataloader: torch.utils.data.DataLoader,
    question_tokenizer: PreTrainedTokenizerFast,
    generator_tokenizer: PreTrainedTokenizerFast,
    k_retrieved: int,
    max_combined_length: int,
    max_answer_length: int,
    device: torch.device,
    epoch_num_for_log="eval",
    max_logged_examples: int = 3, 
    K_DENSE_RETRIEVAL: int = 100,
    wandb_run_obj = None 
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
            original_question_strings = batch["original_question"] 
            current_reference_answers = batch["original_answer"]
            batch_precomputed_sparse_for_eval = batch["precomputed_sparse_docs"]

            # 1. Get Query Embeddings
            query_embeddings_tuple = question_encoder_model(input_ids=q_input_ids, attention_mask=q_attention_mask)
            current_query_embeddings = query_embeddings_tuple[0]

            # 2. Retrieve Documents 
            # For Dense Only
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


def extract_final_answer(text: str) -> str:
    """Extracts the final numeric answer from a GSM8K-style string."""
    match = re.search(r"####\s*([\d\.,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Fallback for generated text: find the last number
    tokens = re.findall(r"-?[\d\.,]+", text)
    return tokens[-1].replace(",", "") if tokens else ""

def evaluate_reasoning(
    generator_model,
    eval_dataloader,
    retriever, # Your DenseRetriever instance
    question_encoder,
    generator_tokenizer,
    k_retrieved,
    device,
    # ... other params
):
    generator_model.eval()
    question_encoder.eval()
    
    predictions = []
    references = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating Reasoning"):
        questions = batch["original_question"]
        gold_solutions = batch["original_solution"]
        
        # 1. Retrieve
        with torch.no_grad():
            q_inputs = retriever.query_encoder.tokenize(questions) # Assuming retriever has tokenizer
            q_inputs = {k: v.to(device) for k,v in q_inputs.items()}
            query_embeddings = question_encoder(**q_inputs)[0]
            retrieved_docs_batch = [retriever.search(q, k=k_retrieved) for q in questions]

        # 2. Construct Prompts and Generate
        prompts = []
        for i, q in enumerate(questions):
            hint_str = "\n".join([f"Hint: {trace['text']}" for trace in retrieved_docs_batch[i]])
            prompt = f"Hint: {hint_str}\nQuestion: {q}\nAnswer:"
            prompts.append(prompt)
            
        inputs = generator_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            output_ids = generator_model.generate(**inputs, max_new_tokens=512)
        
        generated_solutions = generator_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # 3. Extract final answers for comparison
        for gen_sol, gold_sol in zip(generated_solutions, gold_solutions):
            predictions.append(extract_final_answer(gen_sol))
            references.append(extract_final_answer(gold_sol))

    # 4. Calculate Final Answer Accuracy
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    accuracy = correct / len(predictions) if predictions else 0.0
    
    print(f"Final Answer Accuracy: {accuracy:.4f}")
    return {"final_answer_accuracy": accuracy}

def evaluate_rag_sequence(
    question_encoder, 
    generator_model, 
    dense_retriever, 
    eval_dataloader, 
    generator_tokenizer,
    device,
    k_retrieved=10
):

    """
    Runs the full evaluation pipeline for the RAG-Sequence model.
    """
    question_encoder.eval()
    generator_model.eval()

    all_predictions = []
    all_golds = []
    logged_examples = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Running Evaluation"):
            questions = batch['original_question']
            gold_answers = batch['original_answer']

            # Process each question in the batch individually
            for i, question_text in enumerate(questions):
                # 1. Retrieve and Rerank Documents for a single question
                retrieved_docs = dense_retriever.search(question_text, k=50)
                if not retrieved_docs:
                    # Handle cases where no documents are found
                    all_predictions.append("")
                    all_golds.append(extract_gsm8k_gold_answer(gold_answers[i]))
                    continue

                query_doc_pairs = [[question_text, doc['solution_chunk']] for doc in retrieved_docs]
                reranker_scores = dense_retriever.model.compute_score(query_doc_pairs)['colbert']
                for doc, score in zip(retrieved_docs, reranker_scores):
                    doc['rerank_score'] = score
                
                reranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
                top_k_docs = reranked_docs[:k_retrieved]
                
                # 2. Prepare k inputs for the generator, one for each document
                contexts = [doc['solution_chunk'] for doc in top_k_docs]
                inputs_for_gen = [f"Question: {question_text} Context: {c}" for c in contexts]
                
                tokenized_inputs = generator_tokenizer(
                    inputs_for_gen,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                # 3. Generate k answers and get their scores
                generated_outputs = generator_model.generate(
                    **tokenized_inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                sequence_scores = generated_outputs.sequences_scores
                best_sequence_idx = torch.argmax(sequence_scores).item()
                
                # Select the single best answer based on the highest score
                best_generation_ids = generated_outputs.sequences[best_sequence_idx]
                best_generated_answer = generator_tokenizer.decode(best_generation_ids, skip_special_tokens=True)

                # 4. Parse and store results for the question
                pred_ans = extract_final_numeric_answer(best_generated_answer)
                gold_ans = extract_gsm8k_gold_answer(gold_answers[i])
                all_predictions.append(pred_ans)
                all_golds.append(gold_ans)
                
                logged_examples.append({
                    "question": question_text,
                    "gold_answer": gold_answers[i],
                    "parsed_gold": gold_ans,
                    "generated_answer": best_generated_answer,
                    "parsed_prediction": pred_ans,
                    "best_answer_score": sequence_scores[best_sequence_idx].item()
                })

    # --- Final Metric Calculation ---
    accuracy = compute_answer_accuracy(all_predictions, all_golds)
    return {"EM_accuracy": accuracy}, logged_examples

def evaluate_rag_token(
    question_encoder, 
    generator_model, 
    dense_retriever, 
    eval_dataloader, 
    generator_tokenizer,
    device,
    k_retrieved=5
):
    """
    Runs an evaluation pipeline that mimics the RAG-Token style.
    It combines the output probabilities (logits) from multiple documents
    before generating the final token sequence.
    """
    question_encoder.eval()
    generator_model.eval()

    all_predictions = []
    all_golds = []
    logged_examples = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Running Evaluation"):
            questions = batch['original_question'] # Use original strings for pipeline
            gold_answers = batch['original_answer']

            # Process each question in the batch individually
            for i, question_text in enumerate(questions):
                # 1. Retrieve and Rerank Documents for a single question
                retrieved_docs = dense_retriever.search(question_text, k=20)
                if not retrieved_docs:
                    all_predictions.append("")
                    all_golds.append(extract_gsm8k_gold_answer(gold_answers[i]))
                    continue

                query_doc_pairs = [[question_text, doc['solution_chunk']] for doc in retrieved_docs]
                reranker_scores = dense_retriever.model.compute_score(query_doc_pairs)['colbert']
                for doc, score in zip(retrieved_docs, reranker_scores):
                    doc['rerank_score'] = score
                
                reranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
                top_k_docs = reranked_docs[:k_retrieved]
                
                # 2. Prepare k inputs for the generator, one for each document
                contexts = [doc['solution_chunk'] for doc in top_k_docs]
                inputs_for_gen = [f"Question: {question_text} Context: {c}" for c in contexts]
                
                tokenized_inputs = generator_tokenizer(
                    inputs_for_gen,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt"
                ).to(device)

                # 3. Generate logits for each of the k inputs
                # This is a single forward pass that gets all the logits at once.
                outputs = generator_model(**tokenized_inputs)
                
                # outputs.logits has shape [k_retrieved, sequence_length, vocab_size]
                # We average the logits across the k documents to get a single probability distribution
                avg_logits = torch.mean(outputs.logits, dim=0) # Shape: [sequence_length, vocab_size]
                
                # 4. Decode the final answer by taking the most likely token at each position
                predicted_token_ids = torch.argmax(avg_logits, dim=-1)
                
                # Decode the token IDs into a text string
                generated_answer = generator_tokenizer.decode(predicted_token_ids, skip_special_tokens=True)

                # 5. Parse and store results for the question
                pred_ans = extract_final_numeric_answer(generated_answer)
                gold_ans = extract_gsm8k_gold_answer(gold_answers[i])
                all_predictions.append(pred_ans)
                all_golds.append(gold_ans)
                
                logged_examples.append({
                    "question": question_text,
                    "gold_answer": gold_answers[i],
                    "parsed_gold": gold_ans,
                    "generated_answer": generated_answer,
                    "parsed_prediction": pred_ans,
                })

    # --- Final Metric Calculation ---
    accuracy = compute_answer_accuracy(all_predictions, all_golds)
    return {"EM_accuracy": accuracy}, logged_examples

def extract_final_numeric_answer(generated_text: str) -> str:
    """
    Heuristic: Find the last integer or decimal number in the generated text.
    Returns that number as a string, or "" if none found.
    Handles numbers with commas.
    """
    if isinstance(generated_text, str):
        generated_text = generated_text.replace(',', '')
    else:
        generated_text = str(generated_text)
    tokens = re.findall(r"-?\d+\.\d+|-?\d+", generated_text)
    return tokens[-1] if tokens else ""

def extract_gsm8k_gold_answer(answer_field: str) -> str:
    """
    From GSM8K 'answer' field (e.g., "...#### <num>"), extracts the gold numeric answer.
    """
    if isinstance(answer_field, str):
        m = re.search(r"####\s*([-\d\.,]+)", answer_field)
        if m:
            return m.group(1).replace(',', '')
        tokens = re.findall(r"-?\d+\.\d+|-?\d+", answer_field.replace(',', ''))
        return tokens[-1] if tokens else ""
    return str(answer_field)

def compute_answer_accuracy(preds: list, golds: list) -> float:
    """
    Computes exact-match accuracy between two lists of parsed numeric strings.
    """
    if len(preds) != len(golds):
        raise ValueError("Prediction and gold lists must have the same length.")
    correct = sum(1 for p, g in zip(preds, golds) if str(p).strip() == str(g).strip())
    return (correct / len(preds)) * 100 if preds else 0.0