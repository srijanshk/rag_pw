import torch
import re
import json
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import BGEM3FlagModel
from DenseRetriever import DenseRetriever

# --- 1. CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# --- 2. INITIALIZATION & DATA LOADING ---

def initialize_components():
    """Loads all models and the retriever."""
    print("--- Initializing Models and Components ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    components = {}
    
    print("Loading Llama 3.1 model...")
    components['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_NAME)
    components['llm'] = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Set pad_token if it's not already set
    if components['tokenizer'].pad_token is None:
        components['tokenizer'].pad_token = components['tokenizer'].eos_token
        components['llm'].config.pad_token_id = components['llm'].config.eos_token_id
        print("Tokenizer `pad_token` set to `eos_token`.")

    print("Loading Retriever (BGE-M3)...")
    embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    components['retriever'] = DenseRetriever(
        embedding_model=embedding_model,
        index_path="/local00/student/shakya/openmath_bge-m3_hnsw_index",
        metadata_path="/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
        device=device
    )
    
    print("--- All components initialized successfully. ---")
    return components

# --- 3. CORE STATIC RAG WORKFLOW ---

def run_retrieval_tool(query: str, retriever, k_final=3):
    """Executes retrieval and reranking, returns a single context string."""
    retrieved_docs = retriever.search(query, k=20)
    if not retrieved_docs:
        return "No relevant context found."

    query_doc_pairs = [[query, doc['solution_chunk']] for doc in retrieved_docs]
    reranker_scores = retriever.model.compute_score(query_doc_pairs)['colbert']
    for doc, score in zip(retrieved_docs, reranker_scores):
        doc['rerank_score'] = score
    
    reranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
    
    context_text = ""
    for i, doc in enumerate(reranked_docs[:k_final]):
        context_text += f"--- Context Snippet [{i+1}] ---\n{doc['solution_chunk']}\n\n"
    
    return context_text

def get_static_rag_answer(components, question: str):
    """
    Runs the full Static RAG pipeline for a single question.
    Returns both the generated text and the context that was provided.
    """
    llm = components['llm']
    tokenizer = components['tokenizer']
    retriever = components['retriever']
    
    # 1. Retrieve context
    print("\n--- Stage 1: Retrieving and Reranking Context ---")
    retrieved_context = run_retrieval_tool(question, retriever)
    print("✅ Context retrieved.")
    
    # 2. Construct the prompt
    print("--- Stage 2: Constructing Final Prompt ---")
    prompt = (
        "You are an expert mathematician. Use the following context to answer the user's question. Think step-by-step.\n\n"
        f"--- CONTEXT ---\n{retrieved_context}\n"
        f"--- QUESTION ---\n{question}\n\n"
        "--- STEP-BY-STEP ANSWER ---\n"
    )
    print("✅ Prompt constructed.")
    
    # 3. Generate the answer
    print("\n--- Stage 3: Generating Answer from LLM ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    
    outputs = llm.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print("✅ Answer generated.")
    return response_text, retrieved_context

# --- 4. OUTPUT FORMATTING & METRICS ---

def extract_final_numeric_answer(generated_text: str) -> str:
    """
    Heuristic: Find the last integer or decimal number in the generated text.
    """
    tokens = re.findall(r"-?\d+\.\d+|-?\d+", str(generated_text).replace(',', ''))
    return tokens[-1] if tokens else ""


# --- 5. MAIN TEST SCRIPT EXECUTION ---

def main():
    """Main function to run the test on a single example."""
    components = initialize_components()
    
    # A sample question for testing the pipeline
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    print(f"\n{'='*50}\nRUNNING TEST FOR QUESTION:\n'{test_question}'\n{'='*50}")

    generated_answer, retrieved_context = get_static_rag_answer(components, test_question)
    
    print("\n\n--- CONSTRUCTING FINAL FORMATTED OUTPUT ---")

    # The reasoning block now includes the retrieved traces and the LLM's thoughts
    reasoning_block = (
        "--- SEARCHED DATA (TRACES) ---\n"
        f"{retrieved_context.strip()}\n\n"
        "--- LLM REASONING ---\n"
        f"{generated_answer.strip()}"
    )

    # The answer block is the final parsed numeric answer
    answer_block = extract_final_numeric_answer(generated_answer)

    # Construct the final formatted string as requested
    formatted_output = (
        "<reasoning>\n"
        f"{reasoning_block}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{answer_block}\n"
        "</answer>"
    )

    print("\n\n--- FINAL FORMATTED OUTPUT ---")
    print(formatted_output)
    print("\n--- ✅ Test Script Finished ---")


if __name__ == "__main__":
    main()
