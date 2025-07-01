import torch
import re
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import BGEM3FlagModel
from DenseRetriever import DenseRetriever

# --- 1. CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./attention_analysis_results"

# --- 2. INITIALIZATION ---

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
        device_map="auto",
        attn_implementation="eager"
    )

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

# --- 3. WORKFLOWS WITH ATTENTION CAPTURING ---

def get_output_and_attention(components, prompt):
    """
    A generic function to run generation and capture attention scores.
    Returns the full generated sequence and all cross-attentions.
    """
    llm = components['llm']
    tokenizer = components['tokenizer']
    
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    
    with torch.no_grad():
        outputs = llm.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150, # Keep generation shorter for faster analysis
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            output_attentions=True,
            return_dict_in_generate=True
        )

    # The full generated sequence (prompt + new tokens)
    full_sequence = outputs.sequences[0]
    
    # We need the cross-attentions from the decoder
    # Shape: (num_generated_tokens, num_layers, batch, num_heads, 1, context_len)
    attentions = outputs.attentions

    return full_sequence, attentions

# --- Baseline Approach ---
def run_baseline_analysis(components, question):
    prompt = f"Question: {question}\n\nAnswer:"
    return get_output_and_attention(components, prompt)

# --- Static RAG Approach (UPDATED with Reranker) ---
def run_static_rag_analysis(components, question):
    retriever = components['retriever']
    
    # 1. Retrieve a larger set of initial candidates
    retrieved_docs = retriever.search(question, k=50) 
    if not retrieved_docs:
        return None, None
    
    # 2. Rerank the candidates
    query_doc_pairs = [[question, doc['solution_chunk']] for doc in retrieved_docs]
    reranker_scores = retriever.model.compute_score(query_doc_pairs)['colbert']
    for doc, score in zip(retrieved_docs, reranker_scores):
        doc['rerank_score'] = score
    
    reranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
    
    # 3. Use the top-k reranked docs for the context
    context = " ".join([doc['solution_chunk'] for doc in reranked_docs[:3]]) # Use top 3 docs
    
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"--- CONTEXT ---\n{context}\n\n"
        f"--- QUESTION ---\n{question}\n\n"
        "--- STEP-BY-STEP ANSWER ---\n"
    )
    return get_output_and_attention(components, prompt)

# --- Active RAG Approach ---
def run_active_rag_analysis(components, question):
    # For this analysis, we will simulate one turn of active RAG
    # to see how the injected context is used.
    retriever = components['retriever']
    # Simulate a search query the LLM might generate
    simulated_query = f"calculate daily profit from {question.split()[2]} eggs"
    retrieved_docs = retriever.search(simulated_query, k=50)
    if not retrieved_docs:
        return None, None
    
    query_doc_pairs = [[question, doc['solution_chunk']] for doc in retrieved_docs]
    reranker_scores = retriever.model.compute_score(query_doc_pairs)['colbert']
    for doc, score in zip(retrieved_docs, reranker_scores):
        doc['rerank_score'] = score
    
    reranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        
    context = "Search Results:\n" + "\n".join([doc['solution_chunk'] for doc in reranked_docs[:3]])
    
    prompt = (
        f"Question: {question}\n\n"
        f"Okay, I need to figure out how many eggs are left to sell. <search_query>{simulated_query}</search_query>\n\n"
        f"--- Retrieved Context ---\n{context}\n\n"
        f"--- Resuming Answer ---\n"
    )
    return get_output_and_attention(components, prompt)

# --- 4. VISUALIZATION AND SAVING ---

def visualize_and_save_attention_heatmap(full_sequence, attentions, output_path, model_name):
    """
    Visualizes the self-attention as a heatmap and saves the plot.
    """
    if full_sequence is None or attentions is None:
        print(f"Skipping heatmap for {model_name} due to missing data.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Separate prompt from the generated part of the sequence
    num_generated_tokens = len(attentions)
    prompt_len = len(full_sequence) - num_generated_tokens
    
    # Get token labels for the heatmap axes
    input_tokens = tokenizer.convert_ids_to_tokens(full_sequence[:prompt_len])
    generated_tokens = tokenizer.convert_ids_to_tokens(full_sequence[prompt_len:])
    
    # --- FIXED: Process self-attention correctly to create a 2D heatmap ---
    # Create a matrix to store the averaged attention scores
    # Rows: generated tokens, Columns: input tokens
    attention_matrix = torch.zeros(num_generated_tokens, prompt_len)

    # For each generated token, get its attention distribution over the input prompt
    for i in range(num_generated_tokens):
        # `attentions[i]` is a tuple of all layers for the i-th generation step
        step_attentions = attentions[i]
        
        # Stack the attention tensors from all layers for this step
        # Each layer has shape (batch_size, num_heads, 1, context_len_at_this_step)
        step_layers_stacked = torch.stack(step_attentions, dim=0)
        
        # Average over layers and heads, and only take the part that attends to the original prompt
        avg_attention_for_step = step_layers_stacked.mean(dim=(0, 1, 2, 3))[:prompt_len]
        
        # Ensure consistent shape before assignment
        if avg_attention_for_step.shape[0] == prompt_len:
            attention_matrix[i, :] = avg_attention_for_step
        else:
            # Pad if necessary (can happen at the very end of generation)
            padded_attention = torch.zeros(prompt_len)
            padded_attention[:avg_attention_for_step.shape[0]] = avg_attention_for_step
            attention_matrix[i, :] = padded_attention

    # Create DataFrame for heatmap
    df = pd.DataFrame(attention_matrix.cpu().to(torch.float32).numpy(), index=generated_tokens, columns=input_tokens)

    # Plotting
    plt.figure(figsize=(max(18, len(input_tokens) // 2), max(8, len(generated_tokens) // 2)))
    sns.heatmap(df, cmap="viridis", linewidths=.5)
    plt.title(f'Self-Attention Heatmap ({model_name})')
    plt.xlabel("Input Context Tokens")
    plt.ylabel("Generated Output Tokens")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved attention heatmap to {output_path}")

# --- 5. MAIN TEST SCRIPT ---

def main():
    components = initialize_components()
    
    test_questions = {
        "gsm8k_janet": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name, question in test_questions.items():
        print(f"\n{'='*50}\nAnalyzing Question: {name}\n{'='*50}")
        
        # Baseline Run
        print("\n--- Running Baseline Analysis ---")
        base_seq, base_att = run_baseline_analysis(components, question)
        visualize_and_save_attention_heatmap(base_seq, base_att, os.path.join(OUTPUT_DIR, f"{name}_baseline_heatmap.png"), "Baseline")
        
        # Static RAG Run
        print("\n--- Running Static RAG Analysis ---")
        static_seq, static_att = run_static_rag_analysis(components, question)
        visualize_and_save_attention_heatmap(static_seq, static_att, os.path.join(OUTPUT_DIR, f"{name}_static_rag_heatmap.png"), "Static RAG")

        # Active RAG Run
        print("\n--- Running Active RAG Analysis ---")
        active_seq, active_att = run_active_rag_analysis(components, question)
        visualize_and_save_attention_heatmap(active_seq, active_att, os.path.join(OUTPUT_DIR, f"{name}_active_rag_heatmap.png"), "Active RAG")

if __name__ == "__main__":
    main()
