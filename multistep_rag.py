from __future__ import annotations

import logging
import json, os, re, random
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import List, Optional, Dict

import fire, torch
from tqdm import tqdm
from datasets import load_dataset
from rich.progress import track

import faiss 
import numpy as np

from FlagEmbedding import BGEM3FlagModel
from BGERetriever import BGERetriever
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
import wandb 

# ---------------------------------------------------------------------------
# New Multi-Step Prompt Templates
# ---------------------------------------------------------------------------

# Template to force the model to ONLY generate a search query
QUERY_GENERATION_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert at decomposing math word problems into a series of questions. "
    "Given a problem, identify the necessary steps and formulate 1-3 search queries to find the required formulas or methods for each step. "
    "Your queries should be clear questions. Output ONLY the queries, each on a new line."
    "\n<|eot_id|>"
    
    # --- YOUR TASK ---
    "<|start_header_id|>user<|end_header_id|>\n"
    "Problem: {question}"
    "\n<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# Template to generate the final answer using the retrieved context
FINAL_ANSWER_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a mathematical problem solver. Use the user's question and the provided search results to solve the problem. "
    "Explain your reasoning step-by-step. Enclose the final numerical answer in <answer> tags."
    "\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "**Question:**\n{question}\n\n"
    "**Search Results:**\n{retrieved_context}"
    "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

MULTI_QUERY_GENERATION_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert at decomposing math word problems into a series of questions. "
    "Given a problem, identify the necessary steps and formulate 1-3 search queries to find the required formulas or methods for each step. "
    "Your queries should be clear questions. Output ONLY the queries, each on a new line."
    "\n<|eot_id|>"
    
    # --- YOUR TASK ---
    "<|start_header_id|>user<|end_header_id|>\n"
    "Problem: {question}"
    "\n<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


# ---------------------------------------------------------------------------
# Answer extraction helpers (Unchanged)
# ---------------------------------------------------------------------------
CLOSED_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
OPEN_RE   = re.compile(r"<answer>(.*)$", re.S | re.I)
PRE_CLOSE_RE = re.compile( 
    r"([+-]?\d+(?:/\d+)?)(?:\s*|\s*</answer>)",
    re.S | re.I,
)
NUM_RE = re.compile(
    r"(?:"
    r"\\boxed{([^}]*)}"
    r"|\\(?:d)?frac{([^}]*)}{([^}]*)}"
    r"|[+-]?\\d+(?:/\\d+)?"
    r")",
    re.S | re.I,
)
GSM8K_RE = re.compile(r"####\s*([+-]?[0-9,./\\]+)")
CLOSED_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def _clean(num: str):
    if num is None: return None
    return num.strip().lstrip("$€£ ").replace(",", "").strip()

# def _clean(num: str):
#     if num is None:
#         return None
#     num = num.strip()
#     num = re.sub(r"\\boxed{([^}]*)}", r"\\1", num)
#     num = re.sub(r"\\(?:d)?frac{([^}]*)}{([^}]*)}", r"\\1/\\2", num)
#     return num.lstrip("$€£ ").replace(",", "").strip()


def strip_answer(txt: str):
    if not txt:
        return None
    txt_flat = re.sub(r"\\boxed{([^}]*)}", r"\\1", txt)
    txt_flat = re.sub(r"\\(?:d)?frac{([^}]*)}{([^}]*)}", r"\\1/\\2", txt_flat)

    if (m := CLOSED_RE.findall(txt_flat)):
        return _clean(m[-1])
    if (m := OPEN_RE.findall(txt_flat)):
        tail_nums = NUM_RE.findall(m[-1])
        flat = [p for g in tail_nums for p in (g if isinstance(g, tuple) else [g]) if p]
        if flat:
            return _clean(flat[-1])
    if (m := PRE_CLOSE_RE.findall(txt)):
        return _clean(m[-1])
    nums = NUM_RE.findall(txt_flat)
    flat = [p for g in nums for p in (g if isinstance(g, tuple) else [g]) if p]
    return _clean(flat[-1]) if flat else None


# ---------------------------------------------------------------------------
# Custom stopping criterion (Unchanged, still useful for final answer)
# ---------------------------------------------------------------------------
class StopAfterAnswer(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.close_ids = tokenizer("</answer>", add_special_tokens=False).input_ids
        self.open_ids  = tokenizer("<answer>", add_special_tokens=False).input_ids
        self.nl_id     = tokenizer("\n", add_special_tokens=False).input_ids[-1]
        self.sp_id     = tokenizer(" ", add_special_tokens=False).input_ids[-1]

    def _match_tail(self, ids, pattern):
        return ids.shape[1] >= len(pattern) and torch.all(
            ids[0, -len(pattern):] == torch.tensor(pattern, device=ids.device)
        )

    def _answer_has_digit(self, ids) -> bool:
        tail = ids[0, -256:].tolist()
        text = self.tok.decode(tail, skip_special_tokens=True)
        return bool(re.search(r"<answer>[^\\n]*\\d", text, re.S))

    def __call__(self, input_ids, scores, **kwargs):
        if self._match_tail(input_ids, self.close_ids):
            return True
        if (self._match_tail(input_ids, [*self.open_ids, self.nl_id])
                and self._answer_has_digit(input_ids)):
            return True
        if (self._match_tail(input_ids, [self.sp_id])
                and self._answer_has_digit(input_ids)):
            return True
        return False

# ---------------------------------------------------------------------------
# Dataset loader (Unchanged)
# ---------------------------------------------------------------------------
def load_split(path: str, split: str):
    if os.path.exists(path):
        if path.endswith((".json", ".jsonl")):
            return load_dataset("json", data_files={split: path}, split=split)
        if os.path.isdir(path):
            fp = os.path.join(path, f"{split}.jsonl")
            if os.path.isfile(fp):
                return load_dataset("json", data_files={split: fp}, split=split)
            import glob
            files = glob.glob(os.path.join(path, "**", f"{split}.jsonl"), recursive=True)
            if files:
                return load_dataset("json", data_files={split: files}, split=split)
    return load_dataset(path, split=split)

# ---------------------------------------------------------------------------
# Reranking helper (Unchanged)
# ---------------------------------------------------------------------------
def colbert_scores_safe(
    query: str,
    docs: list[dict],
    model: BGEM3FlagModel,
    batch_pairs: int = 16,
) -> list[float]:
    pairs = [[query, d.get("solution_chunk") or d.get("text", "")] for d in docs]
    scores = []
    for i in range(0, len(pairs), batch_pairs):
        try:
            sc = model.compute_score(
                pairs[i:i+batch_pairs],
                batch_size=batch_pairs,                
            )["colbert"]
        except RuntimeError as e:
            logging.warning("ColBERT OOM on %d–%d: %s → use dense",
                            i, i+batch_pairs-1, e)
            sc = [d["dense_score"] for d in docs[i:i+batch_pairs]]
        scores.extend(float(x) for x in sc)
    return scores


def main(
    model_name: str,
    dataset_path: str,
    faiss_index: str,
    split: str,
    out_path: str,
    faiss_meta: str,
    batch: int = 6,
    max_new: int = 1024,
    seed: int = 42,
    search_k: int = 200,
    k_final: int = 5,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    debug: bool = False,
):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_split(dataset_path, split)
    if isinstance(ds[0], str):
        raise ValueError("Dataset rows are strings – supply JSONL records.")
    
    eval_table = None
    if wandb_project:
        wandb.init(
            project=wandb_project, 
            name=wandb_run_name or f"RAG_{Path(dataset_path).stem}_{split}",
            config={ "model": model_name, "dataset": dataset_path, "split": split, "search_k": search_k, "k_final": k_final, "seed": seed }
        )
        eval_table = wandb.Table(columns=[
            "id", "question", "generated_queries", "retrieved_context", 
            "raw_response", "prediction", "ground_truth"
        ])

    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Llama-3 models expect left padding for batched generation
    tok.padding_side = "left"

    multi_gpu = int(os.getenv("WORLD_SIZE", "1")) > 1
    print(f"Using {device} with multi-GPU={multi_gpu}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, # Use bfloat16 for better performance
        device_map="auto",
    )
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.eos_token_id
    
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    model.eval()

    print("Loading retrieval model...")
    retriever_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    retriever = BGERetriever(
        embedding_model=retriever_model,
        index_path=faiss_index,
        metadata_path=faiss_meta,
        device=device,
    )
    
    def rerank(query: str, docs: List[Dict], k: int = 5) -> List[Dict]:
        if not docs:
            return []
        scores = colbert_scores_safe(query, docs, retriever.model, batch_pairs=16)
        for d, s in zip(docs, scores):
            d["rerank_score"] = s
        return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:k]

    final_results = []
    for idx, row in enumerate(track(ds, desc="Generating")):
        question = row.get("question", row.get("problem", ""))
        true_answer_text = row.get("answer", row.get("solution", ""))
        
        # --- STEP 1: Generate a LIST of Search Queries ---
        query_prompt = MULTI_QUERY_GENERATION_TEMPLATE.format(question=question)
        inputs = tok(query_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tok.eos_token_id)
        
        # Split the output into a list of queries
        queries_text = tok.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        generated_queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        
        if debug:
            print(f"\n--- Sample {idx}: {question[:100]}...")
            print(f"Generated Queries:\n{queries_text}")

        # --- STEP 2: Perform Retrieval for EACH query and combine results ---
        all_retrieved_context = []
        unique_docs = {} # Use dict to de-duplicate documents
        if generated_queries:
            for query in generated_queries:
                try:
                    docs = retriever.search(query, k=search_k)
                    top_docs = rerank(query, docs, k=k_final)
                    for doc in top_docs:
                        # Use a unique identifier from your doc to avoid duplicates
                        doc_id = doc.get('id') or doc.get('solution_chunk')
                        if doc_id not in unique_docs:
                            unique_docs[doc_id] = doc
                except Exception as e:
                    print(f"⚠️ Retrieval error for query '{query}': {e}")
        
        # Format the unique, combined context
        retrieved_examples = [f"Example {i+1}: {doc.get('problem', '')[:150]}... Solution: {doc.get('solution_chunk', '')[:200]}..." for i, doc in enumerate(unique_docs.values())]
        retrieved_context_str = "\n".join(retrieved_examples) if retrieved_examples else "No relevant documents found."
        
        # --- STEP 3: Generate Final Answer with the combined context ---
        final_prompt = FINAL_ANSWER_TEMPLATE.format(question=question, retrieved_context=retrieved_context_str)
        inputs = tok(final_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)

        final_solution = tok.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        predicted_answer = strip_answer(final_solution)
        ground_truth_answer = strip_answer(true_answer_text)

        # --- STEP 4: Record Results ---
        result_data = {
            "id": row.get("id", idx), "question": question, "generated_queries": generated_queries,
            "retrieved_context": retrieved_context_str, "raw_response": final_solution,
            "prediction": predicted_answer, "ground_truth": ground_truth_answer,
            "raw_ground_truth": true_answer_text
        }
        final_results.append(result_data)

        if eval_table is not None:
            eval_table.add_data(
                result_data["id"], 
                result_data["question"], 
                "\n".join(result_data["generated_queries"]),
                result_data["retrieved_context"], 
                result_data["raw_response"], 
                result_data["prediction"], 
                result_data["ground_truth"]
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f: json.dump(final_results, f, indent=2)
    print(f"\n✅ Completed. Results saved to: {out_path}")

    if wandb_project:
        print("Logging final results table to W&B...")
        wandb.log({"multi_query_results_table": eval_table})
        wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)