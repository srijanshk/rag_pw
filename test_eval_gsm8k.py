import torch
import pandas as pd
import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from torch.utils.data import Dataset, DataLoader

# --- Step 1: Import all custom components ---
from DenseRetriever import DenseRetriever 
from QuestionEncoder import QuestionEncoder
from FlagEmbedding import BGEM3FlagModel
from MathDataset import MathDataset, load_gsm8k_from_file, collate_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

# --- Step 2: Define Evaluation Helper Functions Locally ---
# These functions are now part of this script for self-containment.
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


# --- Step 3: Define the Core Evaluation Function (Rewritten for RAG) ---
def evaluate_rag_sequence(
    question_encoder, 
    generator_model, 
    dense_retriever, 
    eval_dataloader, 
    question_tokenizer, 
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
                    max_length=1024,
                    return_tensors="pt"
                ).to(device)

                # 3. Generate k answers and get their scores
                generated_outputs = generator_model.generate(
                    **tokenized_inputs,
                    max_length=256,
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
                
                if len(logged_examples) < 10: # Log first 10 examples
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
    question_tokenizer, 
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
                
                if len(logged_examples) < 10: # Log first 10 examples
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



# --- Step 4: Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    QUESTION_ENCODER_NAME = "BAAI/bge-m3"
    GENERATOR_NAME = "facebook/bart-large-cnn"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
    FAISS_INDEX_PATH = "/local00/student/shakya/openmath_bge-m3_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/openmath_bge-m3_metadata.jsonl"
    
    # --- Load Models and Tokenizers ---
    print("--- Loading Pre-Trained Models for Evaluation ---")
    try:
        question_tokenizer = AutoTokenizer.from_pretrained(QUESTION_ENCODER_NAME)
        q_encoder_config = AutoConfig.from_pretrained(QUESTION_ENCODER_NAME)
        question_encoder = QuestionEncoder(config=q_encoder_config, model_name_or_path=QUESTION_ENCODER_NAME).to(DEVICE)

        generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
        generator_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_NAME).to(DEVICE)
        
        embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        dense_retriever = DenseRetriever(
            embedding_model=embedding_model,
            index_path=FAISS_INDEX_PATH,
            metadata_path=METADATA_PATH,
            device=DEVICE
        )
    except OSError as e:
        print(f"\nError loading models. Please check model names and internet connection.")
        print(f"Details: {e}")
        exit()

    # --- Load Data using the imported MathDataset module ---
    print("\n--- Loading Evaluation Data ---")
    eval_data_list = load_gsm8k_from_file(GSM8K_TEST_FILE, limit=20)
    
    eval_dataset = MathDataset(
        eval_data_list, 
        tokenizer=generator_tokenizer,
        max_q_len=128,
        max_a_len=256,
        question_key='question', 
        answer_key='answer'
    )
        
    eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=collate_fn)

    # --- Run Evaluation ---
    metrics, logged_examples = evaluate_rag_sequence(
        question_encoder,
        generator_model,
        dense_retriever,
        eval_loader,
        question_tokenizer,
        generator_tokenizer,
        DEVICE
    )

    # metrics, logged_examples = evaluate_rag_token(
    #     question_encoder,
    #     generator_model,
    #     dense_retriever,
    #     eval_loader,
    #     question_tokenizer,
    #     generator_tokenizer,
    #     DEVICE
    # )

    # --- Display Results ---
    print("\n--- EVALUATION COMPLETE ---")
    print(f"Exact Match Accuracy: {metrics['EM_accuracy']:.2f}%")

    print("\n--- Logged Examples ---")
    results_df = pd.DataFrame(logged_examples)
    print(results_df.to_string())
    
    results_df.to_csv("rag_token_evaluation_outputs.csv", index=False)
    print("\nFull results saved to 'rag_token_evaluation_outputs.csv'")
