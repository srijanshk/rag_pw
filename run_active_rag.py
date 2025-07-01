import torch
import re
import json
import pandas as pd
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

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

# --- 3. ACTIVE RAG WORKFLOW ---

SYSTEM_PROMPT = """You are an expert mathematician. Your goal is to solve the user's question by thinking step-by-step. If you need a formula or fact you don't know, you must use the search tool by writing a query in the format: <search_query>your query here</search_query>. STOP your response after the tag. It will provide search results, and you must use them to continue your reasoning. When you have the final answer, state it clearly."""

def parse_for_search_query(text: str):
    match = re.search(r"<search_query>(.*?)</search_query>", text)
    return match.group(1).strip() if match else None

def run_retrieval_tool(query: str, retriever, k_final=5):
    """Executes retrieval and reranking."""
    retrieved_docs = retriever.search(query, k=50)
    if not retrieved_docs:
        return "No results found."

    query_doc_pairs = [[query, doc['solution_chunk']] for doc in retrieved_docs]
    reranker_scores = retriever.model.compute_score(query_doc_pairs)['colbert']
    for doc, score in zip(retrieved_docs, reranker_scores):
        doc['rerank_score'] = score
    
    reranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
    
    results_text = "Search Results:\n"
    for i, doc in enumerate(reranked_docs[:k_final]):
        results_text += f"[{i+1}] {doc['solution_chunk'][:400]}...\n"
    
    return results_text

def get_active_rag_answer(components, question: str):
    """Runs the active RAG loop and returns the full conversation history."""
    llm = components['llm']
    tokenizer = components['tokenizer']
    retriever = components['retriever']
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    
    for _ in range(5): # Max 5 tool uses per question to prevent infinite loops
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(llm.device)
        
        outputs = llm.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            num_beams=2,  # Use beam search to prevent repetitive output
            early_stopping=True, # Stop when all beams have finished
            output_attentions=True,
            return_dict_in_generate=True
        )
        
        response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text})
        
        search_query = parse_for_search_query(response_text)
        
        if search_query:
            retrieved_results = run_retrieval_tool(search_query, retriever)
            messages.append({"role": "user", "content": retrieved_results})
        else:
            break
            
    return messages

# --- 4. METRIC CALCULATION & OUTPUT FORMATTING ---

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

def format_final_output(conversation_trace: list):
    """Formats the conversation into the desired <reasoning> and <answer> blocks."""
    reasoning_block = ""
    for msg in conversation_trace:
        if msg['role'] == 'system': continue
        
        # Add clear markers for the search queries and responses
        if msg['role'] == 'assistant' and '<search_query>' in msg['content']:
            reasoning_block += f"--- LLM Reasoning & Search Query ---\n{msg['content']}\n\n"
        elif msg['role'] == 'user' and 'Search Results:' in msg['content']:
            reasoning_block += f"--- Retrieved Traces ---\n{msg['content']}\n\n"
        elif msg['role'] == 'user': # The initial user question
             reasoning_block += f"--- Initial Question ---\n{msg['content']}\n\n"
        else: # Final assistant response
            reasoning_block += f"--- LLM Reasoning ---\n{msg['content']}\n\n"
    
    final_assistant_message = conversation_trace[-1]['content']
    answer_block = extract_final_numeric_answer(final_assistant_message)
    
    return (
        f"<reasoning>\n{reasoning_block.strip()}\n</reasoning>\n"
        f"<answer>\n{answer_block}\n</answer>"
    )

# --- 5. MAIN EVALUATION SCRIPT ---

def run_evaluation(components, questions, gold_answers, dataset_name):
    """A generic function to run evaluation on any given dataset."""
    results_data = []
    
    for question, gold_answer in tqdm(zip(questions, gold_answers), total=len(questions), desc=f"Evaluating Active RAG on {dataset_name}"):
        conversation_trace = get_active_rag_answer(components, question)
        final_generated_text = conversation_trace[-1]['content']
        
        parsed_prediction = extract_final_numeric_answer(final_generated_text)
        parsed_gold = extract_gold_answer(gold_answer, dataset_name)
        formatted_output = format_final_output(conversation_trace)
        
        results_data.append({
            "dataset": dataset_name,
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": final_generated_text,
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
        project="active-rag-evaluation",
        name="llama3.1-active-rag-gsm8k-math",
        config={"model_name": MODEL_NAME, "evaluation_type": "active_rag"}
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
    output_filename = "active_rag_results_combined.csv"
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
