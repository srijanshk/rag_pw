import torch
import re
import json
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import BGEM3FlagModel
from DenseRetriever import DenseRetriever

# --- 1. CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
MATH_BASE_DIR = "./thesis_datasets/math_hendrycks"
MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", 
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

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
    
    if components['tokenizer'].pad_token is None:
        components['tokenizer'].pad_token = components['tokenizer'].eos_token
        components['llm'].config.pad_token_id = components['llm'].config.eos_token_id
        print("Tokenizer `pad_token` set to `eos_token`.")
    
    components['llm'].generation_config.temperature = None
    components['llm'].generation_config.top_p = None
    print("Removed default temperature and top_p to suppress warnings.")

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

def load_gsm8k_local(path):
    """Loads GSM8K test data."""
    questions, answers = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            questions.append(obj.get("question", "").strip())
            answers.append(obj.get("answer", "").strip())
    return questions, answers

def load_math_hendrycks_local(base_dir, subjects):
    """Loads MATH test data."""
    questions, answers = [], []
    for subject in subjects:
        file_path = os.path.join(base_dir, subject, "test.jsonl")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    questions.append(obj.get("problem", "").strip())
                    answers.append(obj.get("solution", "").strip())
    return questions, answers

# --- 3. CORE STATIC RAG WORKFLOW ---

def run_retrieval_tool(query: str, retriever, k_final=5):
    """Executes retrieval and reranking, returns a single context string."""
    retrieved_docs = retriever.search(query, k=50)
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
    """Runs the full Static RAG pipeline for a single question."""
    llm = components['llm']
    tokenizer = components['tokenizer']
    retriever = components['retriever']
    
    retrieved_context = run_retrieval_tool(question, retriever)
    
    prompt = (
        "You are an expert mathematician. Use the following context to answer the user's question. Think step-by-step.\n\n"
        f"--- CONTEXT ---\n{retrieved_context}\n"
        f"--- QUESTION ---\n{question}\n\n"
        "--- STEP-BY-STEP ANSWER ---\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    
    outputs = llm.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_beams=2,  # Use beam search to prevent repetitive output
        early_stopping=True, # Stop when all beams have finished
    )
    
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response_text, retrieved_context

# --- 4. EVALUATION, METRICS & FORMATTING ---

def extract_final_numeric_answer(text: str) -> str:
    tokens = re.findall(r"-?\d+\.\d+|-?\d+", str(text).replace(',', ''))
    return tokens[-1] if tokens else ""

def extract_gold_answer(answer_field: str, dataset_name: str):
    """Parses the gold answer based on the dataset format."""
    if dataset_name == "GSM8K":
        m = re.search(r"####\s*([-\d\.,]+)", str(answer_field))
        if m: return m.group(1).replace(',', '')
    elif dataset_name == "MATH":
        m = re.search(r"\\boxed\{(.+?)\}", str(answer_field))
        if m: return extract_final_numeric_answer(m.group(1))
    return extract_final_numeric_answer(str(answer_field))

def compute_accuracy(preds, golds):
    correct = sum(1 for p, g in zip(preds, golds) if p.strip() == str(g).strip())
    return (correct / len(preds)) * 100 if preds else 0.0

def format_static_rag_output(retrieved_context: str, generated_answer: str):
    """Formats the output into the desired <reasoning> and <answer> blocks."""
    reasoning_block = (
        "--- SEARCHED DATA (TRACES) ---\n"
        f"{retrieved_context.strip()}\n\n"
        "--- LLM REASONING ---\n"
        f"{generated_answer.strip()}"
    )
    answer_block = extract_final_numeric_answer(generated_answer)
    return (
        f"<reasoning>\n{reasoning_block}\n</reasoning>\n"
        f"<answer>\n{answer_block}\n</answer>"
    )

# --- 5. MAIN SCRIPT EXECUTION ---

def run_evaluation(components, questions, gold_answers, dataset_name):
    """A generic function to run evaluation on any given dataset."""
    results_data = []
    
    for question, gold_answer in tqdm(zip(questions, gold_answers), total=len(questions), desc=f"Evaluating Static RAG on {dataset_name}"):
        generated_answer, retrieved_context = get_static_rag_answer(components, question)
        
        parsed_prediction = extract_final_numeric_answer(generated_answer)
        parsed_gold = extract_gold_answer(gold_answer, dataset_name)
        formatted_output = format_static_rag_output(retrieved_context, generated_answer)
        
        results_data.append({
            "dataset": dataset_name,
            "question": question,
            "gold_answer": gold_answer,
            "retrieved_context": retrieved_context,
            "generated_answer": generated_answer,
            "parsed_prediction": parsed_prediction,
            "parsed_gold": parsed_gold,
            "formatted_output": formatted_output,
        })

    predictions = [r['parsed_prediction'] for r in results_data]
    golds = [r['parsed_gold'] for r in results_data]
    accuracy = compute_accuracy(predictions, golds)
    
    print(f"\n--- {dataset_name} Evaluation Complete ---")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")
    
    df = pd.DataFrame(results_data)
    return df, accuracy

def main():
    """Main function to run the evaluation on both benchmarks."""
    # --- WandB Setup ---
    wandb.init(
        project="static-rag-evaluation",
        name="llama3.1-static-rag-gsm8k-math",
        config={"model_name": MODEL_NAME, "evaluation_type": "static_rag_math"},
    )
    
    components = initialize_components()
    
    # Run GSM8K evaluation
    print("\n--- Starting GSM8K Evaluation ---")
    gsm8k_questions, gsm8k_answers = load_gsm8k_local(GSM8K_TEST_FILE)
    gsm8k_df, gsm8k_accuracy = run_evaluation(components, gsm8k_questions, gsm8k_answers, "GSM8K")
    
    # Run MATH evaluation
    print("\n--- Starting MATH Evaluation ---")
    math_questions, math_answers = load_math_hendrycks_local(MATH_BASE_DIR, MATH_SUBJECTS)
    math_df, math_accuracy = run_evaluation(components, math_questions, math_answers, "MATH")

    # Combine results and save
    print("\n--- Combining and Saving All Results ---")
    combined_df = pd.concat([gsm8k_df, math_df], ignore_index=True)
    output_filename = "static_rag_results_gsm8k.csv"
    combined_df.to_csv(output_filename, index=False)
    print(f"Full results for both benchmarks saved to {output_filename}")
    
    # --- Log final metrics and tables to WandB ---
    print("\n--- Logging Results to WandB ---")
    wandb.log({
        "GSM8K_Accuracy": gsm8k_accuracy,
        "MATH_Accuracy": math_accuracy,
        "Combined_Results_Table": wandb.Table(dataframe=combined_df)
    })
    
    print("--- Evaluation and Logging Complete. ---")
    wandb.finish()

if __name__ == "__main__":
    main()
