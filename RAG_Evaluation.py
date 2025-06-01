import os

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import traceback
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
import faiss

from NqDataset import NQDataset
from QuestionEncoder import QuestionEncoder
from DenseRetriever import DenseRetriever
from RagUtils import calculate_rag_loss, prepare_generator_inputs, retrieve_documents_for_batch
from utils import load_local_nq_json, custom_collate_fn, load_precomputed_sparse_results
from RagEval import evaluate_custom_rag_model


current_wandb_run = None

def run_evaluation_test():
    print("--- Starting Standalone Evaluation Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using device: {device}")

    retriever_e5_model_name = "models/retriever_finetuned_e5_best"
    question_encoder_path = "rag_train_hybrid_v4/best_model/question_encoder"
    question_tokenizer_path = "rag_train_hybrid_v4/best_model/question_tokenizer"
    generator_tokenizer_path = "rag_train_hybrid_v4/best_model/generator_tokenizer"
    generator_path = "rag_train_hybrid_v4/best_model/bart_generator" 
    
    EVAL_DATA_FILE = "downloads/data/gold_passages_info/nq_dev.json"
    SPARSE_EVAL_FILE = "downloads/data/nq_dev_sparse_retrieval.jsonl" 
    FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
    
    k_dense_for_hybrid = 50
    k_retrieved_for_generator = 50

    MAX_QUESTION_LENGTH = 128
    MAX_ANSWER_LENGTH = 64 
    EVAL_BATCH_SIZE = 4
    MAX_COMBINED_LENGTH_FOR_GEN = 512
    EVAL_DATA_LIMIT = None

    # 1. Initialize Tokenizers
    print(f"Loading E5 question tokenizer from: {question_tokenizer_path}")
    e5_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_path)
    print(f"Loading BART generator tokenizer from: {generator_tokenizer_path}")
    bart_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_path)
    print("Tokenizers initialized.")

    # 2. Initialize Models
    print(f"Loading E5 Question Encoder: {question_encoder_path}")
    question_encoder = QuestionEncoder.from_pretrained(question_encoder_path).to(device)
    question_encoder.eval() 
    print("E5 Question Encoder loaded.")

    print(f"Loading BART Generator: {generator_path}")
    generator = AutoModelForSeq2SeqLM.from_pretrained(generator_path).to(device)
    if hasattr(generator.config, "forced_bos_token_id") and \
       generator.config.forced_bos_token_id is None and \
       generator.config.bos_token_id is not None:
        generator.config.forced_bos_token_id = generator.config.bos_token_id
        print(f"Set generator.config.forced_bos_token_id to {generator.config.bos_token_id}")
    print("BART Generator loaded.")
    
    # 3. Initialize DenseRetriever
    print(f"Initializing custom DenseRetriever with E5: {retriever_e5_model_name}")
    dense_retriever_instance = DenseRetriever(
        FAISS_INDEX_PATH, METADATA_PATH, device, retriever_e5_model_name,
        ef_search=1500, ef_construction=200, fine_tune=False, doc_encoder_model=retriever_e5_model_name)
    print("DenseRetriever initialized.")

    # 4. Load Evaluation Data
    print(f"Loading NQ evaluation data from: {EVAL_DATA_FILE}")
    try:
        eval_data_list = load_local_nq_json(EVAL_DATA_FILE, limit=EVAL_DATA_LIMIT)
        if not eval_data_list:
            print("Evaluation data list is empty. Cannot run evaluation test.")
            return
    except Exception as e:
        print(f"Failed to load evaluation NQ dataset: {e}")
        traceback.print_exc()
        return
    
    eval_sparse_data_lookup = load_precomputed_sparse_results(SPARSE_EVAL_FILE)
    eval_dataset = NQDataset(eval_data_list, eval_sparse_data_lookup, e5_tokenizer, bart_tokenizer, MAX_QUESTION_LENGTH, MAX_ANSWER_LENGTH)
    eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, collate_fn=custom_collate_fn)
    
    if len(eval_dataloader) == 0:
        print("Evaluation DataLoader is empty. Cannot run evaluation test.")
        return
    print(f"Evaluation DataLoader created with {len(eval_dataloader)} batches.")

    # 5. Optional: Initialize WandB for this test if you want to log
    global current_wandb_run # Use the global or pass wandb.run object
    try:
        import wandb
        # You might want a different project/name for standalone eval tests
        final_eval_wandb_run = wandb.init(
                project="rag-final-evaluation", # Or your preferred project
                name="final_best_model_nq_dev_run", # Descriptive name
                reinit=True, # Good for standalone scripts
                config={
                    "eval_data_file": EVAL_DATA_FILE,
                    "sparse_eval_file": SPARSE_EVAL_FILE,
                    "eval_data_limit": EVAL_DATA_LIMIT,
                    "k_retrieved_for_generator": k_retrieved_for_generator,
                    "k_dense_for_hybrid": k_dense_for_hybrid,
                    "max_q_len": MAX_QUESTION_LENGTH,
                    "max_a_len": MAX_ANSWER_LENGTH,
                    "max_combined_len_gen": MAX_COMBINED_LENGTH_FOR_GEN,
                    "eval_batch_size": EVAL_BATCH_SIZE
                }
            )
        current_wandb_run = final_eval_wandb_run # So evaluate_custom_rag_model can use it
        print("WandB initialized for evaluation test.")
    except Exception as e:
        print(f"Wandb could not be initialized for test: {e}")
        current_wandb_run = None


    # 6. Call the evaluation function
    # Ensure all parameters match the definition in custom_rag_eval.py
    eval_metrics, logged_samples = evaluate_custom_rag_model(
        question_encoder_model=question_encoder,
        dense_retriever=dense_retriever_instance,
        generator_model=generator,
        eval_dataloader=eval_dataloader, # This dataloader now yields batches with sparse docs
        question_tokenizer=e5_tokenizer,
        generator_tokenizer=bart_tokenizer,
        k_retrieved=k_retrieved_for_generator, # Final k docs for generator
        max_combined_length=MAX_COMBINED_LENGTH_FOR_GEN,
        max_answer_length=MAX_ANSWER_LENGTH,
        device=device,
        epoch_num_for_log="final_best_model_eval", # Unique identifier for this run's logs
        max_logged_examples=100,
        K_DENSE_RETRIEVAL=k_dense_for_hybrid, 
        wandb_run_obj=current_wandb_run
    )

    print("\n--- Standalone Evaluation Test Finished ---")
    print("Evaluation Metrics:", eval_metrics)
    if logged_samples:
        print("\nLogged Samples (first few):")
        for i, sample in enumerate(logged_samples[:2]): # Print first 2 logged samples
            print(f"Sample {i+1}: {sample}")
            
    if current_wandb_run:
        if logged_samples: # Log table if samples were generated
            try:
                wandb_table = pd.DataFrame(logged_samples)
                current_wandb_run.log({"evaluation_test_examples": wandb.Table(dataframe=wandb_table)})
                print("Logged evaluation examples to WandB.")
            except Exception as e:
                print(f"Error logging evaluation examples to WandB table: {e}")
        current_wandb_run.log(eval_metrics) # Log final metrics
        current_wandb_run.finish()
        print("WandB run finished.")

if __name__ == "__main__":
    run_evaluation_test()

