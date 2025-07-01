import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# --- Import all your custom components ---
# You will need to make sure these files can be imported correctly
from DenseRetriever import DenseRetriever
from test_summarization_layer import summarize_all_docs # Or move this function to RagUtils.py
from test_synthesis_layer import synthesize_trace # Or move this function to RagUtils.py
from baseline_model import load_gsm8k_local # To load your test data

# --- 1. SETUP AND MODEL LOADING ---

print("--- Initializing All Models for RAG Synthesizer ---")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Retriever
print("Loading Retriever...")
dense_retriever = DenseRetriever(
    index_path="/local00/student/shakya/openmath_bge-m3_hnsw_index",
    metadata_path="/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
    device=device,
)
# Note: The retriever model itself (BGE-M3) is loaded inside the DenseRetriever class

# Load Summarizer (Distiller)
print("Loading Summarizer Model (BART)...")
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)

# Load Reasoner (Llama)
print("Loading Reasoner Model (Llama 3.1)...")
reasoner_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b-Instruct")
reasoner_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8b-Instruct",
    torch_dtype=torch.float16,
    device_map="auto" # This will manage GPU placement
)
if reasoner_tokenizer.pad_token is None:
    reasoner_tokenizer.pad_token = reasoner_tokenizer.eos_token

print("--- All models loaded successfully. ---")


# --- 2. LOAD EVALUATION DATASET ---

print("Loading GSM8K test data...")
# We can reuse the loader from your baseline_model.py
gsm_questions, gsm_answers, _ = load_gsm8k_local("./thesis_datasets/gsm8k/test.jsonl")
# Optional: Limit the number of questions for a quick test run
# gsm_questions = gsm_questions[:10]
# gsm_answers = gsm_answers[:10]


# --- 3. MAIN EVALUATION LOOP ---

results = []
print(f"Starting evaluation on {len(gsm_questions)} questions...")

for question, gold_answer in tqdm(zip(gsm_questions, gsm_answers), total=len(gsm_questions)):
    try:
        # --- Step 3a: Retrieval Layer ---
        # Note: The DenseRetriever needs a query embedding, not a string.
        # We'll need a small helper to get the question embedding first.
        # For simplicity, let's assume the retriever class handles this.
        # This part might need adjustment based on your DenseRetriever implementation.
        retrieved_contexts = dense_retriever.retrieve_documents(question, k=5) # Assuming k=5
        
        # --- Step 3b: Summarization Layer ---
        list_of_summaries = summarize_all_docs(retrieved_contexts, summarizer_model, summarizer_tokenizer)
        
        # --- Step 3c: Synthesis Layer ---
        final_prompt_text = synthesize_trace(question, list_of_summaries)
        
        # --- Step 3d: Reasoning Layer ---
        inputs = reasoner_tokenizer(final_prompt_text, return_tensors="pt").to(reasoner_model.device)
        
        with torch.no_grad():
            output_ids = reasoner_model.generate(
                **inputs,
                max_new_tokens=256, # Or a higher value for complex math problems
                do_sample=False,
                pad_token_id=reasoner_tokenizer.pad_token_id
            )
        
        # Decode only the newly generated tokens
        generated_answer = reasoner_tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        
        # --- Store results ---
        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "synthesized_trace": "\n- ".join(list_of_summaries),
            "generated_answer": generated_answer
        })

    except Exception as e:
        print(f"Error processing question: {question}\n{e}")
        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "synthesized_trace": "ERROR",
            "generated_answer": f"ERROR: {e}"
        })


# --- 4. SAVE THE RESULTS ---
print("Evaluation complete. Saving results to CSV...")
df = pd.DataFrame(results)
df.to_csv("synthesizer_model_outputs.csv", index=False)
print("Results saved to synthesizer_model_outputs.csv")