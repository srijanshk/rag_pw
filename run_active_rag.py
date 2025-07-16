import torch
import re
import json
import pandas as pd
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from FlagEmbedding import BGEM3FlagModel
from BGERetriever import BGERetriever


# 1 · Configuration constants

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
STOP_STRING = "</search>"                            # sentinel that pauses gen
MAX_SEARCH_TURNS = 2                                # safety‑guard

GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
MATH_BASE_DIR = "./thesis_datasets/math_hendrycks"
MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", 
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

# 2 · Helper: one‑time construction of StopWordsCriteria

class StopWordsCriteria(StoppingCriteria):
    """Token‑level early‑exit that stops once *any* stop‑sequence appears."""

    def __init__(self, stop_words_ids, tokenizer):
        super().__init__()
        # expect a *list of lists* of ids (same contract as HF impl)
        self.stop_words = [torch.tensor(sw, dtype=torch.long) for sw in stop_words_ids]
        self.stop_len = [len(sw) for sw in self.stop_words]

    def _check_stop(self, input_ids):
        last_n_tokens = input_ids[0]  # shape: (seq_len,)
        for sw, n in zip(self.stop_words, self.stop_len):
            if last_n_tokens.size(0) < n:
                continue
            if torch.equal(last_n_tokens[-n:], sw.to(last_n_tokens.device)):
                return True
        return False

    def __call__(self, input_ids, scores, **kwargs):
        return self._check_stop(input_ids)

def build_stop_criteria(tokenizer, stop_str: str = STOP_STRING) -> StoppingCriteriaList:
    """Return a StoppingCriteriaList that halts as soon as *stop_str* appears."""
    # `tokenizer.encode` returns a *flat* list of ids; StopWordsCriteria expects
    # a *list of token‑id lists*, hence the surrounding `[   ]`.
    stop_ids = [tokenizer.encode(stop_str, add_special_tokens=False)]
    return StoppingCriteriaList([StopWordsCriteria(stop_ids, tokenizer)])

# 3 · System prompt that drives the Active‑RAG loop
SYSTEM_PROMPT = """
You are an expert competition-level mathematician with a large private memory **and**
on-demand access to a math knowledge base. Use retrieval **only when your own recall seems
insufficient.**

### Format you MUST follow
1. **Recall (max 8 sentences)**  
   Think step-by-step and list the theorems / definitions that might solve the problem.  
   Finish this stage with exactly one line:  
   `NEED_SEARCH: yes`    – if you are <80 % confident your recall is enough  
   `NEED_SEARCH: no`     – if you are ≥80 % confident

2. **(optional) Search**  
   If and only if `NEED_SEARCH: yes`, submit one query inside  
   `<search>[your query here]</search>`  
   · Keep it ≤12 tokens, no special characters.  
   · After the search results are shown to you, summarise **at most 5 key facts** you actually use.

3. **Solve**  
   Combine your recall (and retrieved facts if any) into a rigorous derivation.  
   Finish with the answer in the form `$\boxed{\\text{final answer}}$`.

4. **If genuinely unsolved**, reply exactly `I cannot solve this problem.`

### Norms
- Prefer internal recall; retrieval is a *fallback*, not a crutch.  
- Never hallucinate references—quote or paraphrase retrieved snippets instead.  
- Keep your chain-of-thought short, factual, and free of unnecessary speculation.
"""


# 4 · Component initialisation (LLM + retriever)
def initialize_components():
    """Load Llama‑3, BGE‑M3 retriever, tokenizer, etc."""
    print("--- Initialising models and retriever ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- LLM ---------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        llm.config.pad_token_id = tok.eos_token_id

    # Keep generation fully deterministic for evaluation
    llm.generation_config.temperature = None
    llm.generation_config.top_p = None

    # ---- Retriever --------------------------------------------------------
    emb_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    retriever = BGERetriever(
        embedding_model=emb_model,
        index_path="/local00/student/shakya/openmath_bge-m3_hnsw_index",
        metadata_path="/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
        device=device,
    )

    return {"tokenizer": tok, "llm": llm, "retriever": retriever,
            "stopper": build_stop_criteria(tok)}

# 5 ·  Utility: parse search query inside <search> tags

def parse_for_search_query(text: str):
    match = re.search(r"<search>(.*?)</search>", text)
    return match.group(1).strip() if match else None

def need_search(text: str) -> bool:
    """Return True if the assistant decided NEED_SEARCH: yes."""
    m = re.search(r"NEED_SEARCH:\s*(yes|no)", text, re.I)
    return bool(m and m.group(1).lower() == "yes")

# 6 ·  Run one Active‑RAG conversation (may trigger multiple searches)

def get_active_rag_answer(components, question: str):
    tok, llm = components["tokenizer"], components["llm"]
    retriever = components["retriever"]
    stopper = components["stopper"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    search_used = False

    # ----- conversation loop (LLM → maybe search → LLM → …) --------------
    for _turn in range(MAX_SEARCH_TURNS):
        # 1) LLM step -------------------------------------------------------
        input_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt").to(llm.device)
        out = llm.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=1024,
            stopping_criteria=stopper,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=1.15,
            do_sample=False,
        )
        assistant_text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_text})

        # 2) DECISION: NEED_SEARCH? ----------------------------------------
        if not need_search(assistant_text):
            # Model is confident or provided final answer → stop loop
            break

        # 3) RETRIEVAL PATH -------------------------------------------------
        search_query = parse_for_search_query(assistant_text)
        if not search_query:
            # Model claimed it needs search but failed to emit query → ask again
            messages.append({
                "role": "user",
                "content": "You said NEED_SEARCH: yes but gave no <search>. Please provide the query inside <search> tags."
            })
            continue  # go to next turn to let model try again

        search_used = True
        retrieved = run_retrieval_tool(search_query, retriever)

        # Append traces and loop once more ---------------------------------
        messages.append({"role": "user", "content": f"Search Results: {retrieved}"})
    else:
        # Reached MAX_SEARCH_TURNS without final answer
        messages.append({"role": "assistant", "content": "I cannot solve this problem."})

    return messages, search_used

# 7 ·  Retrieval helper 
def run_retrieval_tool(query: str, retriever: BGERetriever, k_final: int = 5):
    """Retrieve → rerank → return nicely formatted text."""
    docs = retriever.search(query, k=50)
    if not docs:
        return "No results found."

    pairs = [[query, d["solution_chunk"]] for d in docs]
    scores = retriever.model.compute_score(pairs)["colbert"]
    for d, s in zip(docs, scores):
        d["rerank_score"] = s
    docs = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:k_final]

    return "Search Results:\n" + "\n".join(
        f"[{i+1}] problem: {d['problem']}\n      solution: {d['solution_chunk']}…"
        for i, d in enumerate(docs)
    )

# 8. Load local datasets (GSM8K and MATH)

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

#9 . METRIC CALCULATION & OUTPUT FORMATTING

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
        
        if msg['role'] == 'assistant' and '<search>' in msg['content']:
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

# 10. MAIN EVALUATION SCRIPT ---

def run_evaluation(components, questions, gold_answers, dataset_name):
    """A generic function to run evaluation on any given dataset."""
    results_data = []
    
    for question, gold_answer in tqdm(zip(questions, gold_answers), total=len(questions), desc=f"Evaluating Active RAG on {dataset_name}"):
        conversation_trace, search_used = get_active_rag_answer(components, question)
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
            "search_used": search_used,
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
        name="llama3.1-active-rag-gsm8k",
        config={"model_name": MODEL_NAME, "evaluation_type": "active_rag"}
    )
    
    components = initialize_components()
    
    # Run GSM8K evaluation
    print("\n--- Starting GSM8K Evaluation ---")
    gsm8k_questions, gsm8k_answers = load_gsm8k_local(GSM8K_TEST_FILE)
    gsm8k_df, gsm8k_accuracy = run_evaluation(components, gsm8k_questions, gsm8k_answers, "GSM8K")
    
    # Run MATH evaluation
    # print("\n--- Starting MATH Evaluation ---")
    # math_questions, math_answers = load_math_hendrycks_local(MATH_BASE_DIR, MATH_SUBJECTS)
    # math_df, math_accuracy = run_evaluation(components, math_questions, math_answers, "MATH")

    # Combine results and save
    print("\n--- Combining and Saving All Results ---")
    combined_df = pd.concat([gsm8k_df], ignore_index=True)
    output_filename = "active_rag_results_gsm8k.csv"
    combined_df.to_csv(output_filename, index=False)
    print(f"Full results for both benchmarks saved to {output_filename}")
    
    # --- Log final metrics and tables to WandB ---
    wandb.log({
        "GSM8K_Accuracy": gsm8k_accuracy,
        # "MATH_Accuracy": math_accuracy,
        "Combined_Results_Table": wandb.Table(dataframe=combined_df)
    })
    
    print("--- Evaluation and Logging Complete. ---")
    wandb.finish()

if __name__ == "__main__":
    main()
