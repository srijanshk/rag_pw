import os

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

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
from utils import load_local_nq_json, custom_collate_fn
from RagEval import evaluate_custom_rag_model, evaluate_dense_rag_model


current_wandb_run = None

def run_retrieval_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration (same as your main.py)
    retriever_e5_model_name = "models/retriever_finetuned_e5_best" # For Q-encoder & Doc-encoder in DenseRetriever
    FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
    k_for_retrieval = 10
    MAX_QUESTION_LENGTH_FOR_TEST = 128

    print(f"Loading E5 tokenizer from: {retriever_e5_model_name}")
    question_tokenizer = AutoTokenizer.from_pretrained(retriever_e5_model_name)

    print(f"Loading E5 question encoder model from: {retriever_e5_model_name}")
    e5_base_config = AutoConfig.from_pretrained(retriever_e5_model_name)
    question_encoder_model = QuestionEncoder(config=e5_base_config, model_name_or_path=retriever_e5_model_name)
    question_encoder_model.to(device)
    question_encoder_model.eval()
    print("E5 Question Encoder model loaded.")

    # 3. Initialize DenseRetriever
    print(f"Initializing DenseRetriever...")

    dense_retriever_instance = DenseRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        device=device,
        model_name=retriever_e5_model_name,
        ef_search=1500,
        ef_construction=200,
        fine_tune=False
    )
    print("DenseRetriever initialized.")

    # 4. Sample Questions
    sample_questions = [
        "What is the capital of France?",
        "Who wrote the novel Moby Dick?",
        "When was the first RAG paper published?"
    ]
    print(f"\nTest Questions: {sample_questions}")

    # 5. Tokenize Questions
    tokenized_batch = question_tokenizer(
        sample_questions,
        max_length=MAX_QUESTION_LENGTH_FOR_TEST,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    q_input_ids = tokenized_batch.input_ids.to(device)
    q_attention_mask = tokenized_batch.attention_mask.to(device)

    # 6. Get Query Embeddings
    print("\nGenerating query embeddings...")
    with torch.no_grad():
        query_embeddings_tuple = question_encoder_model(input_ids=q_input_ids, attention_mask=q_attention_mask)
        query_embeddings = query_embeddings_tuple[0]
    print(f"Query embeddings shape: {query_embeddings.shape}")

    # 7. Retrieve Documents
    print(f"\nRetrieving top {k_for_retrieval} documents...")
    try:
        retrieved_info = retrieve_documents_for_batch(
            query_embeddings_batch=query_embeddings,
            dense_retriever=dense_retriever_instance,
            k=k_for_retrieval,
            normalize_query_for_faiss=True # E5 often uses normalized embeddings
        )
    except Exception as e:
        print(f"ERROR during retrieve_documents_for_batch: {e}")
        traceback.print_exc()
        return

    # 8. Print Results for Inspection
    print("\n--- Retrieval Test Results ---")
    for i in range(len(sample_questions)):
        print(f"\nFor Question: \"{sample_questions[i]}\"")
        print(f"  Retrieved FAISS IDs: {retrieved_info['retrieved_doc_faiss_ids'][i]}")
        print(f"  Retrieved Distances: {retrieved_info['retrieved_doc_faiss_distances'][i]}")
        for j in range(k_for_retrieval):
            title = retrieved_info['retrieved_doc_titles'][i][j]
            text_snippet = retrieved_info['retrieved_doc_texts'][i][j][:250] + "..."
            doc_embedding_sample = retrieved_info['retrieved_doc_embeddings'][i, j, :3].tolist() # First 3 values
            print(f"    Doc {j+1}:")
            print(f"      Title: {title}")
            print(f"      Text Snippet: {text_snippet}")
            print(f"      Embedding Sample: {doc_embedding_sample}")
    print("\n--- Test Finished ---")

def run_custom_pipeline_retrieval_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    retriever_e5_model_name = "models/retriever_finetuned_e5_best"
    generator_bart_model_name = "facebook/bart-base"
    
    TRAIN_FILE = "downloads/data/gold_passages_info/nq_train.json"
    FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
    
    k_for_retrieval = 10
    TEST_BATCH_SIZE = 8
    MAX_QUESTION_LENGTH = 128 # Max length for E5 question encoder
    MAX_ANSWER_LENGTH = 64   # Max length for BART labels (for NQDataset)
    TEST_DATA_LIMIT = 10     # Load only a few samples for this test

    # 1. Initialize Tokenizers
    print(f"Loading E5 question tokenizer from: {retriever_e5_model_name}")
    e5_q_tokenizer = AutoTokenizer.from_pretrained(retriever_e5_model_name)
    print(f"Loading BART generator tokenizer from: {generator_bart_model_name}")
    bart_g_tokenizer = AutoTokenizer.from_pretrained(generator_bart_model_name)

    # 2. Load Data using new NQDataset and DataLoader
    print("Loading NQ data for testing...")
    try:
        test_data_list = load_local_nq_json(TRAIN_FILE, limit=TEST_DATA_LIMIT)
        if not test_data_list:
            print("No data loaded, test cannot proceed.")
            return
    except Exception as e:
        print(f"Failed to load data for test: {e}")
        return

    # Pass the specific tokenizers to NQDataset
    test_dataset = NQDataset(
        test_data_list, 
        question_tokenizer=e5_q_tokenizer, 
        generator_tokenizer=bart_g_tokenizer, # For labels part
        max_question_length=MAX_QUESTION_LENGTH, 
        max_answer_length=MAX_ANSWER_LENGTH
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=TEST_BATCH_SIZE, 
        collate_fn=custom_collate_fn
    )
    print(f"Test DataLoader created.")
    if len(test_dataloader) == 0:
        print("Test DataLoader is empty. Exiting test.")
        return

    # 3. Initialize E5QuestionEncoder model
    print(f"Loading E5 question encoder model from: {retriever_e5_model_name}")
    e5_base_config = AutoConfig.from_pretrained(retriever_e5_model_name)
    question_encoder_model = QuestionEncoder(config=e5_base_config, model_name_or_path=retriever_e5_model_name)
    question_encoder_model.to(device)
    question_encoder_model.eval()
    print("E5 Question Encoder model loaded.")

    # 4. Initialize DenseRetriever
    print(f"Initializing DenseRetriever...")
    dense_retriever_instance = DenseRetriever(
        index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH, device=device,
        model_name=retriever_e5_model_name, fine_tune=False # model_name for its internal doc_encoder
    )
    print("DenseRetriever initialized.")

    # 5. Get a batch from DataLoader
    print("\nFetching a batch from DataLoader...")
    try:
        batch = next(iter(test_dataloader))
    except StopIteration:
        print("Test DataLoader became empty unexpectedly.")
        return
    
    print(f"DEBUG: Keys available in batch: {list(batch.keys())}")    
    q_input_ids = batch["input_ids"].to(device)
    q_attention_mask = batch["attention_mask"].to(device)
    original_questions_in_batch = batch["original_question"] # Available if needed

    print(f"Processing a batch of {q_input_ids.shape[0]} questions...")

    # 6. Get Query Embeddings
    print("Generating query embeddings for the batch...")
    with torch.no_grad():
        query_embeddings_tuple = question_encoder_model(input_ids=q_input_ids, attention_mask=q_attention_mask)
        current_query_embeddings = query_embeddings_tuple[0]
    print(f"Generated query embeddings shape: {current_query_embeddings.shape}")

    # 7. Call retrieve_documents_for_batch
    print(f"\nRetrieving top {k_for_retrieval} documents per query...")
    try:
        retrieved_batch_info = retrieve_documents_for_batch(
            query_embeddings_batch=current_query_embeddings,
            dense_retriever=dense_retriever_instance,
            k=k_for_retrieval,
            normalize_query_for_faiss=True 
        )
        print("retrieve_documents_for_batch function called successfully.")
    except Exception as e:
        print(f"ERROR during retrieve_documents_for_batch: {e}")
        traceback.print_exc()
        return

    # 8. Print the outputs for inspection
    print("\n--- Output from retrieve_documents_for_batch ---")
    print(f"Keys in output: {list(retrieved_batch_info.keys())}")

    retrieved_doc_embeddings = retrieved_batch_info.get('retrieved_doc_embeddings')
    retrieved_doc_faiss_ids = retrieved_batch_info.get('retrieved_doc_faiss_ids')
    retrieved_doc_faiss_distances = retrieved_batch_info.get('retrieved_doc_faiss_distances')
    retrieved_doc_texts_batch = retrieved_batch_info.get('retrieved_doc_texts')
    retrieved_doc_titles_batch = retrieved_batch_info.get('retrieved_doc_titles')

    if retrieved_doc_embeddings is not None:
        print(f"\nShape of retrieved_doc_embeddings: {retrieved_doc_embeddings.shape}")
        # Expected: [TEST_BATCH_SIZE, k_for_retrieval, doc_embedding_dimension]
    else:
        print("\n'retrieved_doc_embeddings' not found in output.")
    
    if retrieved_doc_faiss_ids is not None:
        print(f"Shape of retrieved_doc_faiss_ids: {retrieved_doc_faiss_ids.shape}")
    else:
        print("'retrieved_doc_faiss_ids' not found in output.")
        
    if retrieved_doc_faiss_distances is not None:
        print(f"Shape of retrieved_doc_faiss_distances: {retrieved_doc_faiss_distances.shape}")
    else:
        print("'retrieved_doc_faiss_distances' not found in output.")


    if retrieved_doc_texts_batch and len(retrieved_doc_texts_batch) > 0:
        print("\nRetrieved document details (first item in batch):")
        for batch_item_idx in range(min(len(original_questions_in_batch), len(retrieved_doc_texts_batch))):
            print(f"\n  For Original Question: \"{original_questions_in_batch[batch_item_idx]}\"")
            if batch_item_idx < len(retrieved_doc_texts_batch):
                for doc_idx in range(len(retrieved_doc_texts_batch[batch_item_idx])): # Loop through k docs for this query
                    title = retrieved_doc_titles_batch[batch_item_idx][doc_idx] if retrieved_doc_titles_batch else "N/A"
                    text_snippet = (retrieved_doc_texts_batch[batch_item_idx][doc_idx][:200] + "...") if retrieved_doc_texts_batch else "N/A"
                    faiss_id = retrieved_doc_faiss_ids[batch_item_idx][doc_idx] if retrieved_doc_faiss_ids is not None else "N/A"
                    distance = retrieved_doc_faiss_distances[batch_item_idx][doc_idx] if retrieved_doc_faiss_distances is not None else "N/A"
                    
                    if isinstance(distance, (float, np.float32, np.float64)):
                        formatted_distance = f"{distance:.4f}"
                    else:
                        formatted_distance = str(distance) 
                        
                    doc_embedding_sample = "N/A"
                    if retrieved_doc_embeddings is not None and retrieved_doc_embeddings.numel() > 0: 
                        if batch_item_idx < retrieved_doc_embeddings.shape[0] and doc_idx < retrieved_doc_embeddings.shape[1]:
                           doc_embedding_sample = retrieved_doc_embeddings[batch_item_idx, doc_idx, :3].tolist()

                    # Use the formatted_distance in the print statement
                    print(f"    Doc {doc_idx+1} (FAISS ID: {faiss_id}, Distance: {formatted_distance}):")
                    print(f"      Title: {title}")
                    print(f"      Text Snippet: {text_snippet}")
                    print(f"      Embedding Sample: {doc_embedding_sample}")
            else:
                print("No documents retrieved for this item in batch, or batch was empty.")
    else:
         print("\nNo 'retrieved_doc_texts' found in output or it's empty.")
    
    print("\n--- Retrieval Pipeline Test Finished ---")

# Run the retrieval test
def run_custom_pipeline_gen_input_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using device: {device}")

    # Configuration
    retriever_e5_model_name = "models/retriever_finetuned_e5_best"
    generator_bart_model_name = "best_bart_model" 
    
    NQ_DATA_FILE_FOR_TEST = "downloads/data/gold_passages_info/nq_dev.json" 
    FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
    
    k_retrieved_for_test = 10 
    TEST_BATCH_SIZE = 2    
    MAX_QUESTION_LENGTH_TEST = 128
    MAX_ANSWER_LENGTH_TEST = 64 
    TEST_DATA_LIMIT = TEST_BATCH_SIZE 
    MAX_COMBINED_LENGTH_FOR_GEN = 512 # Or 1024 if your BART can handle it and you prefer

    # 1. Initialize Tokenizers
    print(f"Loading E5 question tokenizer from: {retriever_e5_model_name}")
    e5_q_tokenizer = AutoTokenizer.from_pretrained(retriever_e5_model_name)
    print(f"Loading BART generator tokenizer from: {generator_bart_model_name}")
    bart_g_tokenizer = AutoTokenizer.from_pretrained(generator_bart_model_name)
    print("Tokenizers initialized.")

    # 2. Load Data (NQDataset should return "original_question")
    print("Loading a small sample of NQ data for testing...")
    test_data_list = load_local_nq_json(NQ_DATA_FILE_FOR_TEST, limit=TEST_DATA_LIMIT)
    if not test_data_list: print("No data loaded."); return
        
    test_dataset = NQDataset(
        test_data_list, 
        question_tokenizer=e5_q_tokenizer, 
        generator_tokenizer=bart_g_tokenizer, 
        max_question_length=MAX_QUESTION_LENGTH_TEST, 
        max_answer_length=MAX_ANSWER_LENGTH_TEST
    )
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=custom_collate_fn)
    if len(test_dataloader) == 0: print("Test DataLoader is empty."); return
    print(f"Test DataLoader created.")

    # 3. Initialize E5QuestionEncoder & DenseRetriever (as in previous test)
    e5_base_config = AutoConfig.from_pretrained(retriever_e5_model_name)
    question_encoder_model = QuestionEncoder(config=e5_base_config, model_name_or_path=retriever_e5_model_name)
    question_encoder_model.to(device).eval()
    dense_retriever_instance = DenseRetriever(
        index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH, device=device,
        model_name=retriever_e5_model_name, fine_tune=False
    )
    print("Question Encoder and DenseRetriever initialized.")

    # 4. Get a batch
    batch = next(iter(test_dataloader))
    q_input_ids = batch["input_ids"].to(device)
    q_attention_mask = batch["attention_mask"].to(device)
    original_questions_in_batch = batch["original_question"] 
    print(f"\nProcessing a batch of {len(original_questions_in_batch)} questions...")

    # 5. Get Query Embeddings
    with torch.no_grad():
        query_embeddings = question_encoder_model(input_ids=q_input_ids, attention_mask=q_attention_mask)[0]
    print(f"Generated query embeddings shape: {query_embeddings.shape}")

    # 6. Retrieve Documents
    retrieved_info = retrieve_documents_for_batch(
        query_embeddings_batch=query_embeddings,
        dense_retriever=dense_retriever_instance,
        k=k_retrieved_for_test,
        normalize_query_for_faiss=True
    )
    print("Documents retrieved.")
    # You can add prints here for retrieved_info if you want to double-check it

    # 7. Prepare Generator Inputs using the new function
    print(f"\nPreparing generator inputs (max_combined_length: {MAX_COMBINED_LENGTH_FOR_GEN})...")
    try:
        generator_inputs = prepare_generator_inputs(
            original_question_strings=original_questions_in_batch,
            retrieved_doc_titles=retrieved_info["retrieved_doc_titles"],
            retrieved_doc_texts=retrieved_info["retrieved_doc_texts"],
            generator_tokenizer=bart_g_tokenizer,
            max_combined_length=MAX_COMBINED_LENGTH_FOR_GEN,
            device=device
        )
        print("Generator inputs prepared successfully.")
    except Exception as e:
        print(f"ERROR during prepare_generator_inputs: {e}")
        traceback.print_exc()
        return

    # 8. Print and Inspect Generator Inputs
    print("\n--- Output from prepare_generator_inputs ---")
    print(f"Keys: {list(generator_inputs.keys())}")
    g_input_ids = generator_inputs["generator_input_ids"]
    g_attention_mask = generator_inputs["generator_attention_mask"]

    print(f"Shape of generator_input_ids: {g_input_ids.shape}")
    # Expected: [TEST_BATCH_SIZE * k_retrieved_for_test, MAX_COMBINED_LENGTH_FOR_GEN]
    print(f"Shape of generator_attention_mask: {g_attention_mask.shape}")

    print("\nSample of generator_input_ids (first formatted input, first 50 tokens):")
    if g_input_ids.numel() > 0:
        print(g_input_ids[0, :50]) 
        print("\nDecoded sample generator input (first formatted input):")
        # Ensure skip_special_tokens=False to see BOS/EOS if they are part of the input
        print(bart_g_tokenizer.decode(g_input_ids[0], skip_special_tokens=False)) 
    else:
        print("generator_input_ids is empty.")

    print("\n--- Generator Input Prep Test Finished ---")


def run_custom_pipeline_full_forward_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using device: {device}")

    # Configuration
    retriever_e5_model_name = "models/retriever_finetuned_e5_best"
    generator_bart_model_name = "best_bart_model" 
    
    NQ_DATA_FILE_FOR_TEST = "downloads/data/gold_passages_info/nq_train.json" 
    FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
    
    k_retrieved_for_test = 3 
    TEST_BATCH_SIZE = 2    # Keep small for testing
    MAX_QUESTION_LENGTH_TEST = 128
    MAX_ANSWER_LENGTH_TEST = 64 
    TEST_DATA_LIMIT = TEST_BATCH_SIZE 
    MAX_COMBINED_LENGTH_FOR_GEN = 512

    # 1. Initialize Tokenizers
    print(f"Loading E5 question tokenizer from: {retriever_e5_model_name}")
    e5_q_tokenizer = AutoTokenizer.from_pretrained(retriever_e5_model_name)
    print(f"Loading BART generator tokenizer from: {generator_bart_model_name}")
    bart_g_tokenizer = AutoTokenizer.from_pretrained(generator_bart_model_name)
    print("Tokenizers initialized.")

    # 2. Load Data
    print("Loading a small sample of NQ data for testing...")
    test_data_list = load_local_nq_json(NQ_DATA_FILE_FOR_TEST, limit=TEST_DATA_LIMIT)
    if not test_data_list: print("No data loaded."); return
        
    test_dataset = NQDataset(
        test_data_list, 
        question_tokenizer=e5_q_tokenizer, 
        generator_tokenizer=bart_g_tokenizer, 
        max_question_length=MAX_QUESTION_LENGTH_TEST, 
        max_answer_length=MAX_ANSWER_LENGTH_TEST
    )
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=custom_collate_fn)
    if len(test_dataloader) == 0: print("Test DataLoader is empty."); return
    print(f"Test DataLoader created.")

    # 3. Initialize Models
    e5_base_config = AutoConfig.from_pretrained(retriever_e5_model_name)
    question_encoder_model = QuestionEncoder(config=e5_base_config, model_name_or_path=retriever_e5_model_name)
    question_encoder_model.to(device).eval()

    dense_retriever_instance = DenseRetriever(
        index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH, device=device,
        model_name=retriever_e5_model_name, fine_tune=False
    )
    
    generator_model_test = AutoModelForSeq2SeqLM.from_pretrained(generator_bart_model_name)
    generator_model_test.to(device)
    # For loss calculation, it should be in train() mode if it has dropout, etc.
    # But for just testing the forward pass of loss, eval() is fine if we don't backprop.
    # Let's use train() as it mimics the training scenario better for loss calculation.
    generator_model_test.train() 
    print("Question Encoder, DenseRetriever, and Generator Model initialized.")

    # 4. Get a batch
    batch = next(iter(test_dataloader))
    q_input_ids = batch["input_ids"].to(device)
    q_attention_mask = batch["attention_mask"].to(device)
    original_question_strings_batch = batch["original_question"]
    target_labels_batch = batch["labels"].to(device) # Tokenized answers from NQDataset

    print(f"\nProcessing a batch of {len(original_question_strings_batch)} questions...")

    # 5. Get Query Embeddings
    with torch.no_grad(): # Query embeddings are fixed for this forward pass test if QE not trained yet
        query_embeddings_tuple = question_encoder_model(input_ids=q_input_ids, attention_mask=q_attention_mask)
        current_query_embeddings = query_embeddings_tuple[0]
    print(f"Generated query embeddings shape: {current_query_embeddings.shape}")

    # 6. Retrieve Documents
    retrieved_info = retrieve_documents_for_batch(
        query_embeddings_batch=current_query_embeddings,
        dense_retriever=dense_retriever_instance,
        k=k_retrieved_for_test,
        normalize_query_for_faiss=True
    )
    batch_retrieved_doc_embeddings = retrieved_info["retrieved_doc_embeddings"]
    print("Documents retrieved.")

    # 7. Prepare Generator Inputs (using the function you just tested)
    print(f"\nPreparing generator inputs (max_combined_length: {MAX_COMBINED_LENGTH_FOR_GEN})...")
    generator_inputs = prepare_generator_inputs(
        original_question_strings=original_question_strings_batch,
        retrieved_doc_titles=retrieved_info["retrieved_doc_titles"],
        retrieved_doc_texts=retrieved_info["retrieved_doc_texts"],
        generator_tokenizer=bart_g_tokenizer,
        max_combined_length=MAX_COMBINED_LENGTH_FOR_GEN,
        device=device
    )
    batch_generator_input_ids = generator_inputs["generator_input_ids"]
    batch_generator_attention_mask = generator_inputs["generator_attention_mask"]
    print("Generator inputs prepared successfully.")

    print("\n--- Output from prepare_generator_inputs ---")
    print(f"Keys: {list(generator_inputs.keys())}")
    print(f"Shape of generator_input_ids: {batch_generator_input_ids.shape}")
    print(f"Shape of generator_attention_mask: {batch_generator_attention_mask.shape}")
    if batch_generator_input_ids.numel() > 0:
        print("\nSample of generator_input_ids (first formatted input, first 50 tokens):")
        print(batch_generator_input_ids[0, :50])
        print("\nDecoded sample generator input (first formatted input):")
        print(bart_g_tokenizer.decode(batch_generator_input_ids[0], skip_special_tokens=False))
    else:
        print("generator_input_ids is empty.")
    print("\n--- Generator Input Prep Inspected ---")
    # --- End of your requested print block ---

    # 8. Calculate RAG Loss
    print("\nCalculating RAG loss for the batch...")
    try:
        # For loss calculation, internal layers like dropout should be active if training.
        generator_model_test.train() 
        question_encoder_model.train() # Also set Q Encoder to train mode if its params are part of loss path

        loss = calculate_rag_loss(
            query_embeddings=current_query_embeddings,
            retrieved_doc_embeddings=batch_retrieved_doc_embeddings,
            generator_input_ids=batch_generator_input_ids,
            generator_attention_mask=batch_generator_attention_mask,
            target_labels=target_labels_batch,
            generator_model=generator_model_test,
            generator_pad_token_id=bart_g_tokenizer.pad_token_id,
            n_docs=k_retrieved_for_test,
            device=device
        )
        print(f"\nCalculated RAG Loss: {loss.item()}")
    except Exception as e:
        print(f"ERROR during RAG loss calculation: {e}")
        traceback.print_exc()
        return
        
    print("\n--- Custom RAG Forward Pass & Loss Test Finished ---")

def run_evaluation_test():
    print("--- Starting Standalone Evaluation Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using device: {device}")

    # Configuration
    retriever_e5_model_name = "models/retriever_finetuned_e5_best"
    generator_bart_model_name = "best_bart_model" 
    retriever_tokenizer_path = "rag_train_hybrid_v3/best_model/question_tokenizer"
    generator_tokenizer_path = "rag_train_hybrid_v3/best_model/generator_tokenizer"
    retriever_encoder_path = "rag_train_hybrid_v3/best_model/question_encoder"
    generator_model_path = "rag_train_hybrid_v3/best_model/generator"
    
    EVAL_DATA_FILE = "downloads/data/gold_passages_info/nq_dev.json" 
    FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
    
    k_retrieved = 10
    MAX_QUESTION_LENGTH = 128
    MAX_ANSWER_LENGTH = 64 
    EVAL_BATCH_SIZE = 8
    MAX_COMBINED_LENGTH_FOR_GEN = 512
    EVAL_DATA_LIMIT = None

    # 1. Initialize Tokenizers
    print(f"Loading E5 question tokenizer from: {retriever_tokenizer_path}")
    e5_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_path)
    print(f"Loading BART generator tokenizer from: {generator_tokenizer_path}")
    bart_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_path)
    print("Tokenizers initialized.")

    # 2. Initialize Models
    print(f"Loading E5 Question Encoder: {retriever_encoder_path}")
    # e5_config = AutoConfig.from_pretrained(retriever_encoder_path)
    question_encoder = QuestionEncoder.from_pretrained(retriever_encoder_path).to(device)
    question_encoder.eval() 
    # question_encoder = QuestionEncoder(config=e5_config).to(device)
    print("E5 Question Encoder loaded.")

    print(f"Loading BART Generator: {generator_model_path}")
    generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_path).to(device)
    if hasattr(generator.config, "forced_bos_token_id") and \
       generator.config.forced_bos_token_id is None and \
       generator.config.bos_token_id is not None:
        generator.config.forced_bos_token_id = generator.config.bos_token_id
        print(f"Set generator.config.forced_bos_token_id to {generator.config.bos_token_id}")
    print("BART Generator loaded.")
    
    # 3. Initialize DenseRetriever
    print(f"Initializing custom DenseRetriever with E5: {retriever_e5_model_name}")
    dense_retriever_instance = DenseRetriever(
        FAISS_INDEX_PATH, METADATA_PATH, device, model_name=retriever_e5_model_name,
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
        
    eval_dataset = NQDataset(eval_data_list, None, e5_tokenizer, bart_tokenizer, MAX_QUESTION_LENGTH, MAX_ANSWER_LENGTH)
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
        test_wandb_run = wandb.init(project="rag-custom-eval-test", name="standalone_eval_run", reinit=True, config={
             "eval_limit": EVAL_DATA_LIMIT, "k_retrieved": k_retrieved, 
             "max_q_len": MAX_QUESTION_LENGTH, "max_a_len": MAX_ANSWER_LENGTH,
             "max_combined_len": MAX_COMBINED_LENGTH_FOR_GEN
        })
        current_wandb_run = test_wandb_run # So evaluate_custom_rag_model can use it
        print("WandB initialized for evaluation test.")
    except Exception as e:
        print(f"Wandb could not be initialized for test: {e}")
        current_wandb_run = None


    # 6. Call the evaluation function
    # Ensure all parameters match the definition in custom_rag_eval.py
    eval_metrics, logged_samples = evaluate_dense_rag_model(
        question_encoder_model=question_encoder,
        dense_retriever=dense_retriever_instance,
        generator_model=generator,
        eval_dataloader=eval_dataloader,
        question_tokenizer=e5_tokenizer,
        generator_tokenizer=bart_tokenizer,
        k_retrieved=k_retrieved,
        max_combined_length=MAX_COMBINED_LENGTH_FOR_GEN,
        max_answer_length=MAX_ANSWER_LENGTH, # Pass this
        device=device,
        epoch_num_for_log="test_run",
        max_logged_examples=3, # From evaluate_custom_rag_model default
        wandb_run_obj=current_wandb_run # Pass the wandb run object
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