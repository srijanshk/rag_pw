import os
import torch
import json
import traceback
from tqdm import tqdm
import numpy as np
import faiss


# Set the visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

# Import from your existing project files
from DenseRetriever import DenseRetriever
from QuestionEncoder import QuestionEncoder
from RagUtils import retrieve_documents_for_batch
from utils import custom_collate_fn # Assuming this is available
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from MathDataset import load_gsm8k_data, MathDataset 
# --- Configuration ---
# Paths to your datasets and models
GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
DOC_ENCODER_PATH = "models/retriever_finetuned_e5_best"  # Base model for the retriever
QUESTION_ENCODER_PATH = "models/retriever_finetuned_e5_best"
TOKENIZER_PATH = "models/retriever_finetuned_e5_best"

# Paths to your NEW OpenMath reasoning index
FAISS_INDEX_PATH = "/local00/student/shakya/openmath_hnsw_index"
METADATA_PATH = "/local00/student/shakya/openmath_metadata.jsonl"

# --- Script Parameters ---
# How many GSM8K questions to test (set to None to run all)
TEST_LIMIT = 20
# How many reasoning chunks to retrieve for each question
K_RETRIEVED = 5
# Process questions in batches for efficiency
BATCH_SIZE = 4

def run_gsm8k_retrieval_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer and Question Encoder Model
    # This follows your template exactly.
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    question_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    print(f"Loading Question Encoder from: {QUESTION_ENCODER_PATH}")
    e5_base_config = AutoConfig.from_pretrained(QUESTION_ENCODER_PATH)
    question_encoder_model = QuestionEncoder(config=e5_base_config, model_name_or_path=QUESTION_ENCODER_PATH).to(device)
    question_encoder_model.eval()
    print("✅ Question Encoder model loaded.")

    # 2. Initialize DenseRetriever with the OpenMath Index
    print("Initializing DenseRetriever with OpenMath index...")
    dense_retriever_instance = DenseRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        device=device,
        model_name=DOC_ENCODER_PATH, # Base model for the retriever
        doc_encoder_model=DOC_ENCODER_PATH, # Explicitly set the doc encoder model
        fine_tune=False
    )
    print("✅ Retriever ready.")

    # 3. Load GSM8K Questions
    print(f"Loading {TEST_LIMIT or 'all'} questions from {GSM8K_TEST_FILE}...")
    gsm8k_questions = load_gsm8k_questions(GSM8K_TEST_FILE, limit=TEST_LIMIT)
    
    # 4. Process questions in batches
    print(f"\n--- Starting Retrieval for {len(gsm8k_questions)} GSM8K Questions ---")
    
    for i in tqdm(range(0, len(gsm8k_questions), BATCH_SIZE), desc="Processing Batches"):
        batch_questions = gsm8k_questions[i:i+BATCH_SIZE]
        
        # 5. Tokenize the batch
        tokenized_batch = question_tokenizer(
            batch_questions,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        q_input_ids = tokenized_batch.input_ids.to(device)
        q_attention_mask = tokenized_batch.attention_mask.to(device)

        # 6. Get Query Embeddings for the batch
        with torch.no_grad():
            query_embeddings = question_encoder_model(input_ids=q_input_ids, attention_mask=q_attention_mask)[0]

        # 7. Retrieve Documents for the batch using your RagUtils function
        try:
            retrieved_info = retrieve_documents_for_batch(
                query_embeddings_batch=query_embeddings,
                dense_retriever=dense_retriever_instance,
                k=K_RETRIEVED,
                normalize_query_for_faiss=True # E5 models use normalized embeddings
            )
        except Exception as e:
            print(f"ERROR during retrieve_documents_for_batch: {e}")
            traceback.print_exc()
            continue

        # 8. Print Results for Inspection for each question in the batch
        for j, question in enumerate(batch_questions):
            print("\n" + "="*80)
            print(f"QUERY (GSM8K Question):")
            print(question)
            print("="*80)

            print(f"\n---> Top {K_RETRIEVED} Retrieved Reasoning Chunks (from OpenMath):")
            if not retrieved_info or not retrieved_info['retrieved_doc_texts'][j]:
                print("    No results found.")
                continue

            for rank in range(K_RETRIEVED):
                title = retrieved_info['retrieved_doc_titles'][j][rank]
                text_snippet = retrieved_info['retrieved_doc_texts'][j][rank]
                distance = retrieved_info['retrieved_doc_faiss_distances'][j][rank]

                print(f"\n  [Rank {rank+1}] (Distance: {distance:.4f})")
                print(f"    Retrieved Chunk: {text_snippet}")
            print("-" * 80)

    print("\n✅ Retrieval test complete.")

def run_custom_pipeline_retrieval_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    retriever_e5_model_name = "models/retriever_finetuned_e5_best"
    generator_bart_model_name = "facebook/bart-base"

    TRAIN_FILE = "./thesis_datasets/gsm8k/test.jsonl"
    FAISS_INDEX_PATH = "/local00/student/shakya/openmath_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/openmath_metadata.jsonl"

    k_for_retrieval = 10
    TEST_BATCH_SIZE = 8
    MAX_QUESTION_LENGTH = 128 
    MAX_ANSWER_LENGTH = 512   
    TEST_DATA_LIMIT = 10    


    # 1. Initialize Tokenizers
    print(f"Loading E5 question tokenizer from: {retriever_e5_model_name}")
    e5_q_tokenizer = AutoTokenizer.from_pretrained(retriever_e5_model_name)
    print(f"Loading BART generator tokenizer from: {generator_bart_model_name}")
    bart_g_tokenizer = AutoTokenizer.from_pretrained(generator_bart_model_name)

    # 2. Load Data using new NQDataset and DataLoader
    print("Loading NQ data for testing...")
    try:
        test_data_list = load_gsm8k_data(TRAIN_FILE, limit=TEST_DATA_LIMIT)
        if not test_data_list:
            print("No data loaded, test cannot proceed.")
            return
    except Exception as e:
        print(f"Failed to load data for test: {e}")
        return
    
    test_dataset = MathDataset(
        data_list=test_data_list,
        retriever_tokenizer=e5_q_tokenizer,
        generator_tokenizer=bart_g_tokenizer,
        max_question_length=MAX_QUESTION_LENGTH,
        max_solution_length=MAX_ANSWER_LENGTH
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
        doc_encoder_model=retriever_e5_model_name,
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




if __name__ == "__main__":
    run_custom_pipeline_retrieval_test()