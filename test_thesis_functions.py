import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

import torch
import textwrap
import numpy as np
import json

from FlagEmbedding import BGEM3FlagModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig


# --- Step 1: Import your custom DenseRetriever class ---
from DenseRetriever import DenseRetriever
from MathDataset import MathDataset, load_gsm8k_from_file, collate_fn
from QuestionEncoder import QuestionEncoder
from RagUtils import calculate_rag_loss, retrieve_documents_for_batch, prepare_generator_inputs, calculate_rag_token_loss
from utils import load_local_nq_json, custom_collate_fn


GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
MATH_BASE_DIR   = "./thesis_datasets/math_hendrycks"
MATH_SUBJECTS   = [
    "algebra", "counting_and_probability", "geometry", 
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

def test_retrieval_and_reranking():
    """
    This function tests the full "retrieve and rerank" pipeline.
    """
    # --- Step 2: Setup and Model Loading ---
    print("--- Initializing Models for Retrieval and Reranking Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the BGE-M3 embedding and reranking model
    print("Loading BGE-M3 model...")
    # This model will be used for both creating query embeddings and for reranking
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    
    # Initialize DenseRetriever with the pre-loaded BGE model
    print("Initializing DenseRetriever...")
    dense_retriever = DenseRetriever(
        embedding_model=model,
        index_path="/local00/student/shakya/openmath_bge-m3_hnsw_index",
        metadata_path="/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
        device=device
    )

    print("--- All models loaded successfully. ---")

    # --- Step 3: Define a test question and parameters ---
    test_question = "What is the number of integer solutions to the equation x^2 - y^2 = 256?"
    initial_retrieval_k = 50  # Fetch a larger pool of candidates initially
    final_reranked_k = 10     # The number of docs we want after reranking

    print(f"\n--- Running Test for Question: '{test_question}' ---")

    # --- Step 4: Initial Dense Retrieval (Candidate Fetching) ---
    print(f"\n--- 1. Dense Retrieval Layer (Fetching Top {initial_retrieval_k} Candidates) ---")
    retrieved_docs_meta = dense_retriever.search(test_question, k=initial_retrieval_k)
    print(f"✅ Initial retrieval successful. Found {len(retrieved_docs_meta)} documents.")
    print("Top 5 initial results (before reranking):")
    for i, doc in enumerate(retrieved_docs_meta[:5]):
        print(f"  Rank {i+1} (ID {doc['id']}, FAISS Score: {doc['score']:.4f}): {doc['solution_chunk'][:100]}...")

    # --- Step 5: Reranking Layer ---
    print(f"\n--- 2. Reranking Layer (Scoring {len(retrieved_docs_meta)} candidates) ---")
    
    # BGE-M3's reranker expects a list of pairs: [query, document_text]
    query_doc_pairs = [[test_question, doc['solution_chunk']] for doc in retrieved_docs_meta]
    
    # Use the model's 'compute_score' method for reranking
    reranker_scores = model.compute_score(query_doc_pairs, batch_size=4) # Use a small batch size

    # The scores are a dictionary, we need the 'colbert' or other relevant scores
    # Let's assume we use the 'colbert' score for reranking as it's often effective.
    # We could also use a combination if needed.
    final_scores = reranker_scores['colbert']

    # Add the new reranker score to our retrieved documents' metadata
    for i in range(len(retrieved_docs_meta)):
        retrieved_docs_meta[i]['rerank_score'] = final_scores[i]

    # Sort the documents based on the new reranker score in descending order
    reranked_docs = sorted(retrieved_docs_meta, key=lambda x: x['rerank_score'], reverse=True)

    print(f"✅ Reranking successful.")

    # --- Step 6: Display Final Results ---
    print(f"\n--- FINAL Top {final_reranked_k} Results After Reranking ---")
    wrapper = textwrap.TextWrapper(width=100)
    for i, doc in enumerate(reranked_docs[:final_reranked_k]):
        print(f"\n  Final Rank {i+1}:")
        print(f"    - Document ID: {doc['id']}")
        print(f"    - Initial FAISS Score: {doc['score']:.4f}")
        print(f"    - Final Rerank Score: {doc['rerank_score']:.4f}")
        print(f"    - Text: {wrapper.fill(doc['solution_chunk'])}")
        
    print("\n\n--- ✅ Retrieval System Test Finished Successfully! ---")

def test_data_loader_from_files():
    """
    This function tests the MathDataset class by loading a small, limited
    number of samples from the actual GSM8K and MATH dataset files.
    """
    # --- Step 1: Define Helper Functions to Load Data from Files ---
    def load_gsm8k_from_file(path, limit=None):
        """Loads data from a GSM8K JSONL file, returning a list of dicts."""
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data_list.append(json.loads(line))
        return data_list

    def load_math_from_dir(base_dir, subjects, limit=None):
        """Loads data from the Hendrycks MATH dataset directories."""
        data_list = []
        for subject in subjects:
            if limit and len(data_list) >= limit:
                break
            file_path = os.path.join(base_dir, subject, "test.jsonl")
            if not os.path.exists(file_path):
                print(f"Warning: File not found, skipping: {file_path}")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if limit and len(data_list) >= limit:
                        break
                    data_list.append(json.loads(line))
        return data_list


    # --- Step 2: Define the MathDataset Class (Unchanged) ---
    class MathDataset(Dataset):
        def __init__(self, data_list, tokenizer, max_q_len, max_a_len,
                     question_key='problem', answer_key='solution'):
            self.data = data_list
            self.tokenizer = tokenizer
            self.max_q_len = max_q_len
            self.max_a_len = max_a_len
            self.question_key = question_key
            self.answer_key = answer_key

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            question = sample.get(self.question_key, '')
            answer = sample.get(self.answer_key, '')
            question_encoding = self.tokenizer(question, max_length=self.max_q_len, padding='max_length', truncation=True, return_tensors='pt')
            labels = self.tokenizer(text_target=answer, max_length=self.max_a_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
            return {
                'input_ids': question_encoding['input_ids'].squeeze(0),
                'attention_mask': question_encoding['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0),
                'original_question': question,
                'original_answer': answer
            }

    # --- Step 3: Setup Tokenizer and Load Data from Files ---
    print("--- Initializing Tokenizer and Loading Data from Files ---")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Load a small number of samples for testing
    test_limit = 10
    print(f"Loading up to {test_limit} samples from each dataset file...")
    
    try:
        gsm8k_data_list = load_gsm8k_from_file(GSM8K_TEST_FILE, limit=test_limit)
        math_data_list = load_math_from_dir(MATH_BASE_DIR, MATH_SUBJECTS, limit=test_limit)
    except FileNotFoundError as e:
        print(f"\nERROR: A dataset file was not found. Please check your paths.")
        print(f"Details: {e}")
        return

    print("--- Data loaded. Starting tests. ---")

    # --- Step 4: Test GSM8K Data Loading ---
    print("\n--- 1. Testing: GSM8K Data Format from File ---")
    if gsm8k_data_list:
        gsm8k_dataset = MathDataset(
            gsm8k_data_list, tokenizer, max_q_len=128, max_a_len=128,
            question_key='question', answer_key='answer'
        )
        gsm8k_loader = DataLoader(gsm8k_dataset, batch_size=2)
        
        gsm8k_batch = next(iter(gsm8k_loader))
        print(f"✅ GSM8K data loaded successfully from file ({len(gsm8k_data_list)} samples).")
        print("Batch Keys:", list(gsm8k_batch.keys()))
        print("Shape of 'input_ids':", gsm8k_batch['input_ids'].shape)
        print("Original Question [0]:", gsm8k_batch['original_question'][0][:80] + "...")
    else:
        print("⚠️ No GSM8K data was loaded. Skipping test.")
    
    # --- Step 5: Test MATH Data Loading ---
    print("\n--- 2. Testing: MATH Data Format from Files ---")
    if math_data_list:
        # Using default keys: 'problem' and 'solution'
        math_dataset = MathDataset(math_data_list, tokenizer, max_q_len=128, max_a_len=256)
        math_loader = DataLoader(math_dataset, batch_size=1)

        math_batch = next(iter(math_loader))
        print(f"✅ MATH data loaded successfully from files ({len(math_data_list)} samples).")
        print("Batch Keys:", list(math_batch.keys()))
        print("Shape of 'input_ids':", math_batch['input_ids'].shape)
        print("Original Question [0]:", math_batch['original_question'][0][:80] + "...")
    else:
        print("⚠️ No MATH data was loaded. Skipping test.")

    print("\n\n--- ✅ Data Loader Test Finished Successfully! ---")


def summarize_text(text_to_summarize, model, tokenizer):
    """Generates a summary for a single piece of text."""
    inputs = tokenizer.batch_encode_plus([text_to_summarize], max_length=1024, return_tensors='pt', truncation=True).to(model.device)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, min_length=20, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

def run_summarization_layer(docs, model, tokenizer):
    """Processes a list of documents through the summarizer."""
    summaries = []
    for doc_text in docs:
        summary = summarize_text(doc_text, model, tokenizer)
        summaries.append(summary)
    return summaries

def run_synthesis_layer(question, summaries):
    """Synthesizes the final prompt from the question and summaries."""
    synthesized_context = "\n".join(f"- {s}" for s in summaries)
    final_prompt = (
        f"Question: {question}\n\n"
        f"Based on the following synthesized context, think step by step to provide a detailed answer.\n\n"
        f"Synthesized Context:\n{synthesized_context}\n\n"
        f"Step-by-step Solution:"
    )
    return final_prompt

def test_full_synthesizer_flow_with_full_dataset():
    # --- Step 4: Setup and Model Loading ---
    print("--- Initializing Models for Live Data Integration Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    dense_retriever = DenseRetriever(
        embedding_model=bge_model,
        index_path="/local00/student/shakya/openmath_bge-m3_hnsw_index",
        metadata_path="/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
        device=device
    )
    summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
    print("--- All models loaded successfully. ---")

    # --- Step 5: Load a few examples using the full MathDataset class ---
    print(f"\n--- Loading test data from {GSM8K_TEST_FILE} ---")
    try:
        live_data_list = load_gsm8k_from_file(GSM8K_TEST_FILE, limit=3)
        if not live_data_list:
            print("No data loaded. Exiting test.")
            return
        
        # Use the full MathDataset class
        test_dataset = MathDataset(
            live_data_list, 
            tokenizer=summarizer_tokenizer, # The tokenizer is needed now
            max_q_len=128, 
            max_a_len=256,
            question_key='question',
            answer_key='answer'
        )
        # We need a collate function to handle batching of dictionaries
        def collate_fn(batch):
            return {key: [d[key] for d in batch] for key in batch[0]}

        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        print(f"✅ Loaded {len(test_dataset)} samples for testing.")
    except FileNotFoundError:
        print(f"ERROR: Test file not found at {GSM8K_TEST_FILE}. Please check the path.")
        return

    # --- Step 6: Execute the full pipeline for each test sample ---
    for i, batch in enumerate(test_loader):
        # The batch is now a dictionary of lists. We extract the first item.
        test_question = batch['original_question'][0]
        
        print(f"\n\n{'='*80}")
        print(f"--- RUNNING FULL PIPELINE FOR TEST SAMPLE {i+1} ---")
        print(f"QUESTION: '{test_question}'")
        print(f"{'='*80}")
        
        # Pipeline steps remain the same
        print("\n--- 1. Retrieval & Reranking Layer ---")
        initial_k, final_k = 50, 10
        retrieved_docs_meta = dense_retriever.search(test_question, k=initial_k)
        query_doc_pairs = [[test_question, doc['solution_chunk']] for doc in retrieved_docs_meta]
        reranker_scores = bge_model.compute_score(query_doc_pairs)['colbert']
        for doc, score in zip(retrieved_docs_meta, reranker_scores):
            doc['rerank_score'] = score
        reranked_docs = sorted(retrieved_docs_meta, key=lambda x: x['rerank_score'], reverse=True)
        final_doc_texts = [doc['solution_chunk'] for doc in reranked_docs[:final_k]]
        print(f"✅ Retrieved and reranked. Selected top {len(final_doc_texts)} documents.")
        
        print("\n--- 2. Summarization Layer ---")
        list_of_summaries = run_summarization_layer(final_doc_texts, summarizer_model, summarizer_tokenizer)
        print(f"✅ Summarization successful.")
        
        print("\n--- 3. Synthesis Layer ---")
        final_prompt = run_synthesis_layer(test_question, list_of_summaries)
        print("✅ Synthesis successful.")
        
        print("\n--- FINAL PROMPT READY FOR GENERATOR ---")
        print(textwrap.fill(final_prompt, width=100))

    print(f"\n\n{'='*80}")
    print("--- ✅ Live Data Integration Test Finished Successfully! ---")

def test_distiller_output_quality():
    """
    This function performs a focused test on the summarizer/distiller model
    to analyze the quality of its output and tune generation parameters.

    It follows a clear "start-to-finish" flow for this specific component:
    1.  Setup: Load the pre-trained summarization model and tokenizer.
    2.  Input: Define a sample document that simulates what our retriever would find.
    3.  Process: Run the model to generate a summary of the document.
    4.  Inspect: Print the original and the summary to evaluate the output quality.
    """
    # --- Step 1: Setup and Model Loading ---
    print("--- Initializing Distiller Model for Output Quality Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We use the pre-trained summarizer for this test. After fine-tuning (in a later task),
    # you can point this path to your own fine-tuned model directory.
    MODEL_NAME = "facebook/bart-large-cnn"
    
    print(f"Loading model and tokenizer from: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        print("--- Model and tokenizer loaded successfully. ---")
    except Exception as e:
        print(f"Error: Could not load model. Please check model name and internet connection. Details: {e}")
        return

    # --- Step 2: Define a Representative Input Document ---
    # This text is representative of a detailed document chunk from the OpenMath knowledge base.
    print("\n--- Defining a sample input document... ---")
    sample_document_text = """
    The problem asks for the number of integer solutions to the equation x^2 - y^2 = 256.
    This equation can be factored as a difference of squares: (x - y)(x + y) = 256.
    Let A = x - y and B = x + y. So, A * B = 256. Since x and y are integers, A and B must also be integers.
    Furthermore, we can express x and y in terms of A and B. Adding the two equations (A=x-y, B=x+y) gives 2x = A + B, so x = (A + B) / 2.
    Subtracting the first from the second gives 2y = B - A, so y = (B - A) / 2.
    For x and y to be integers, (A + B) and (B - A) must both be even. This implies that A and B must have the same parity (both even or both odd).
    Since their product A * B = 256 is an even number, it's impossible for both A and B to be odd.
    Therefore, both A and B must be even integers.
    The task now is to find the number of pairs of even integer factors (A, B) of 256.
    The factors of 256 (which is 2^8) are 2^0, 2^1, ..., 2^8. We also have their negative counterparts.
    The integer factors of 256 are: +/-1, +/-2, +/-4, +/-8, +/-16, +/-32, +/-64, +/-128, +/-256. There are 9 positive factors and 9 negative factors, for a total of 18 integer factors.
    We need to find pairs (A,B) where both are even. The even factors are +/-2, +/-4, ..., +/-256. This gives 8 positive even factors and 8 negative even factors, total of 16 even factors.
    For every even factor A, B is uniquely determined as B = 256/A. Since 256 is a power of 2, if A is an even integer, B will also be an even integer.
    So, there are 16 possible choices for A, and each choice gives a valid corresponding B. This means there are 16 integer solutions (x,y).
    """

    # --- Step 3: Process the Input to Generate a Summary ---
    print("--- Generating summary from the document... ---")
    
    # Tokenize the document text
    inputs = tokenizer.batch_encode_plus(
        [sample_document_text],
        max_length=1024, # BART's max context size
        return_tensors='pt',
        truncation=True
    ).to(device)

    # Use the model's generate function. You can tune these parameters.
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,      # Higher beams, more thorough search
        max_length=120,   # Increase for more detail
        min_length=40,    # Increase to avoid overly short summaries
        early_stopping=True
    )

    # Decode the token IDs back into a readable string
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # --- Step 4: Inspect the Results ---
    print("\n" + "="*50)
    print("INSPECTION RESULTS")
    print("="*50)
    
    print("\n--- ORIGINAL DOCUMENT ---")
    print(textwrap.fill(sample_document_text, width=100))
    
    print("\n--- GENERATED SUMMARY (TRACE) ---")
    print(textwrap.fill(summary, width=100))
    print(f"\nSummary Length: {len(summary.split())} words")

    print("\n--- ✅ Generator Output Test Finished Successfully! ---")


def test_rag_sequence_forward_pass():
    """
    This function tests the full forward pass for a single batch of data
    for the end-to-end RAG-Sequence model, using the new DenseRetriever class
    and loading real data with the MathDataset class.
    """
    # --- Step 3: Setup and Model Loading ---
    print("--- Initializing All Models for RAG-Sequence Forward Pass Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    RETRIEVER_ENCODER_NAME = "BAAI/bge-m3"
    GENERATOR_NAME = "facebook/bart-large-cnn"
    FAISS_INDEX_PATH = "/local00/student/shakya/openmath_bge-m3_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/openmath_bge-m3_metadata.jsonl"
    GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
    
    # Load Models and Tokenizers
    print("Loading models and tokenizers...")
    embedding_model = BGEM3FlagModel(RETRIEVER_ENCODER_NAME, use_fp16=True)
    
    print(f"Loading QuestionEncoder with weights from {RETRIEVER_ENCODER_NAME}...")
    q_encoder_config = AutoConfig.from_pretrained(RETRIEVER_ENCODER_NAME)
    question_encoder = QuestionEncoder(config=q_encoder_config, model_name_or_path=RETRIEVER_ENCODER_NAME).to(device)
    
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_NAME).to(device)
    question_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_ENCODER_NAME)
    generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    
    question_encoder.train()
    generator_model.train()

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        device=device
    )
    print("--- All components initialized successfully. ---")

    # --- Step 4: Load a real data batch using MathDataset ---
    print("\n--- Loading a batch of test data from file... ---")
    batch_size = 2
    initial_k = 20 # Fetch more candidates for the reranker
    final_k = 5      # The final number of docs to use after reranking
    
    try:
        live_data_list = load_gsm8k_from_file(GSM8K_TEST_FILE, limit=batch_size)
        test_dataset = MathDataset(
            live_data_list, 
            tokenizer=question_tokenizer,
            max_q_len=128, 
            max_a_len=256,
            question_key='question',
            answer_key='answer'
        )
        def collate_fn(batch_items):
            return {key: [d[key] for d in batch_items] for key in batch_items[0]}
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        batch = next(iter(test_loader))
        print(f"✅ Loaded a batch of {len(batch['original_question'])} samples.")
    except Exception as e:
        print(f"ERROR: Could not load data. Details: {e}")
        return

    # --- Step 5: Run the RAG-Sequence Pipeline ---
    print("\n--- Running RAG-Sequence forward pass... ---")
    
    # 1. Get Query Embeddings
    tokenized_questions = question_tokenizer(batch['original_question'], padding=True, truncation=True, return_tensors='pt').to(device)
    query_embeddings = question_encoder(
        input_ids=tokenized_questions['input_ids'],
        attention_mask=tokenized_questions['attention_mask']
    )[0]

    # 2. Retrieve Document IDs and Texts (initial retrieval)
    retrieved_docs_batch = dense_retriever.search_batch(batch['original_question'], k=initial_k)
    
    # --- ADDED: Reranking Step ---
    print(f"\n--- Reranking top {initial_k} candidates down to {final_k}... ---")
    reranked_batch = []
    for i, question in enumerate(batch['original_question']):
        query_doc_pairs = [[question, doc['solution_chunk']] for doc in retrieved_docs_batch[i]]
        reranker_scores = embedding_model.compute_score(query_doc_pairs)['colbert']
        
        for doc, score in zip(retrieved_docs_batch[i], reranker_scores):
            doc['rerank_score'] = score
            
        reranked_docs = sorted(retrieved_docs_batch[i], key=lambda x: x['rerank_score'], reverse=True)
        reranked_batch.append(reranked_docs[:final_k])
    
    # Use the reranked documents for the next steps
    retrieved_docs_batch = reranked_batch
    # --- END Reranking Step ---

    # 3. Get Document Embeddings for the Loss Function
    batch_doc_texts, batch_doc_titles, flat_doc_texts_to_encode = [], [], []
    for retrieved_docs in retrieved_docs_batch:
        texts = [doc['solution_chunk'] for doc in retrieved_docs]
        titles = [doc.get('problem', 'N/A') for doc in retrieved_docs]
        batch_doc_texts.append(texts)
        batch_doc_titles.append(titles)
        flat_doc_texts_to_encode.extend(texts)
    
    doc_embedding_dict = embedding_model.encode(
        flat_doc_texts_to_encode, return_dense=True, return_sparse=False, return_colbert_vecs=False
    )
    doc_embeddings_flat = doc_embedding_dict['dense_vecs']
    embedding_dim = doc_embeddings_flat.shape[1]
    retrieved_doc_embeddings = torch.from_numpy(doc_embeddings_flat).view(
        batch_size, final_k, embedding_dim
    ).to(device)
    retrieved_doc_embeddings = retrieved_doc_embeddings.to(query_embeddings.dtype)

    # 4. Prepare Generator Inputs
    generator_inputs = prepare_generator_inputs(
        original_question_strings=batch['original_question'],
        retrieved_doc_titles=batch_doc_titles,
        retrieved_doc_texts=batch_doc_texts,
        generator_tokenizer=generator_tokenizer,
        max_combined_length=1024,
        device=device
    )

    # 5. Tokenize target labels
    tokenized_labels = generator_tokenizer(
        text_target=batch['original_answer'], padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    ).to(device)

    # 6. Calculate the joint RAG Loss
    print("\n--- Calculating RAG loss... ---")
    loss = calculate_rag_loss(
        query_embeddings=query_embeddings,
        retrieved_doc_embeddings=retrieved_doc_embeddings,
        generator_input_ids=generator_inputs['generator_input_ids'],
        generator_attention_mask=generator_inputs['generator_attention_mask'],
        target_labels=tokenized_labels.input_ids,
        generator_model=generator_model,
        generator_pad_token_id=generator_tokenizer.pad_token_id,
        n_docs=final_k, # Use final_k here
        device=device
    )
    
    print(f"\n--- ✅ Forward Pass Successful! ---")
    print(f"Retrieved doc embedding shape: {retrieved_doc_embeddings.shape}")
    print(f"Calculated RAG-Sequence Loss for the batch: {loss.item():.4f}")
    print("\n--- ✅ RAG-Sequence Forward Pass Test Finished Successfully! ---")


def test_rag_token_forward_pass():
    """
    This function tests the full forward pass for a single batch of data
    for an end-to-end RAG-Token style model.
    """
    # --- Step 3: Setup and Model Loading ---
    print("--- Initializing All Models for RAG-Token Forward Pass Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    RETRIEVER_ENCODER_NAME = "BAAI/bge-m3"
    GENERATOR_NAME = "facebook/bart-large-cnn"
    FAISS_INDEX_PATH = "/local00/student/shakya/openmath_bge-m3_hnsw_index"
    METADATA_PATH = "/local00/student/shakya/openmath_bge-m3_metadata.jsonl"
    GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
    
    print("Loading models and tokenizers...")
    embedding_model = BGEM3FlagModel(RETRIEVER_ENCODER_NAME, use_fp16=True)
    q_encoder_config = AutoConfig.from_pretrained(RETRIEVER_ENCODER_NAME)
    question_encoder = QuestionEncoder(config=q_encoder_config, model_name_or_path=RETRIEVER_ENCODER_NAME).to(device)
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_NAME).to(device)
    question_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_ENCODER_NAME)
    generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    
    question_encoder.train()
    generator_model.train()

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        device=device
    )
    print("--- All components initialized successfully. ---")

    # --- Step 4: Load a real data batch ---
    print("\n--- Loading a batch of test data from file... ---")
    batch_size = 2
    initial_k = 50
    final_k = 10 
    
    live_data_list = load_gsm8k_from_file(GSM8K_TEST_FILE, limit=batch_size)
    test_dataset = MathDataset(
        live_data_list, tokenizer=generator_tokenizer, max_q_len=128, max_a_len=256,
        question_key='question', answer_key='answer'
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    batch = next(iter(test_loader))
    print(f"✅ Loaded a batch of {len(batch['original_question'])} samples.")

    # --- Step 5: Run the RAG Pipeline to prepare for loss calculation ---
    print("\n--- Running RAG forward pass... ---")
    
    # 1. Get Query Embeddings
    tokenized_questions = question_tokenizer(batch['original_question'], padding=True, truncation=True, return_tensors='pt').to(device)
    query_embeddings = question_encoder(
        input_ids=tokenized_questions['input_ids'],
        attention_mask=tokenized_questions['attention_mask']
    )[0]

    # 2. Retrieve Document IDs and Texts (initial retrieval)
    retrieved_docs_batch = dense_retriever.search_batch(batch['original_question'], k=initial_k)
    
    # --- Reranking Step ---
    print(f"\n--- Reranking top {initial_k} candidates down to {final_k}... ---")
    reranked_batch = []
    for i, question in enumerate(batch['original_question']):
        query_doc_pairs = [[question, doc['solution_chunk']] for doc in retrieved_docs_batch[i]]
        reranker_scores = embedding_model.compute_score(query_doc_pairs)['colbert']
        
        for doc, score in zip(retrieved_docs_batch[i], reranker_scores):
            doc['rerank_score'] = score
            
        reranked_docs = sorted(retrieved_docs_batch[i], key=lambda x: x['rerank_score'], reverse=True)
        reranked_batch.append(reranked_docs[:final_k])
    
    retrieved_docs_batch = reranked_batch
    
    # 3. Get Document Embeddings
    batch_doc_texts, batch_doc_titles, flat_doc_texts_to_encode = [], [], []
    for retrieved_docs in retrieved_docs_batch:
        texts = [doc['solution_chunk'] for doc in retrieved_docs]
        titles = [doc.get('problem', 'N/A') for doc in retrieved_docs]
        batch_doc_texts.append(texts)
        batch_doc_titles.append(titles)
        flat_doc_texts_to_encode.extend(texts)
    
    doc_embedding_dict = embedding_model.encode(
        flat_doc_texts_to_encode, return_dense=True, return_sparse=False, return_colbert_vecs=False
    )
    doc_embeddings_flat = doc_embedding_dict['dense_vecs']
    embedding_dim = doc_embeddings_flat.shape[1]
    retrieved_doc_embeddings = torch.from_numpy(doc_embeddings_flat).view(
        batch_size, final_k, embedding_dim
    ).to(device)
    retrieved_doc_embeddings = retrieved_doc_embeddings.to(query_embeddings.dtype)
        
    # 4. Prepare Generator Inputs
    generator_inputs = prepare_generator_inputs(
        original_question_strings=batch['original_question'],
        retrieved_doc_titles=batch_doc_titles,
        retrieved_doc_texts=batch_doc_texts,
        generator_tokenizer=generator_tokenizer,
        max_combined_length=1024,
        device=device
    )

    # 5. Get Generator Logits
    outputs = generator_model(
        input_ids=generator_inputs['generator_input_ids'],
        attention_mask=generator_inputs['generator_attention_mask']
    )

    # 6. Calculate the custom RAG-Token Loss
    print("\n--- Calculating RAG-Token loss... ---")
    loss = calculate_rag_token_loss(
        query_embeddings=query_embeddings, # Pass query embeddings
        retrieved_doc_embeddings=retrieved_doc_embeddings, # Pass doc embeddings
        generator_outputs=outputs,
        target_labels=batch['labels'].to(device),
        generator_pad_token_id=generator_tokenizer.pad_token_id,
        n_docs=final_k,
        device=device
    )
    
    print(f"\n--- ✅ Forward Pass Successful! ---")
    print(f"Calculated RAG-Token Loss for the batch: {loss.item():.4f}")
    print("\n--- ✅ RAG-Token Forward Pass Test Finished Successfully! ---")


if __name__ == "__main__":
    try:
        test_rag_token_forward_pass()
    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
    

