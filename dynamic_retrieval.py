from __future__ import annotations

import logging
import json, os, re, random
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
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
# Llama‑3.1 prompt
# ---------------------------------------------------------------------------
RETRIEVE_TRIGGER = "<search>"
SEARCH_CLOSE     = "</search>"
RETRIEVED_START  = "<retrieved>"
RETRIEVED_END    = "</retrieved>"

# Enhanced system message to encourage retrieval
SYSTEM_MSG = """
You are a mathematical problem solver who MUST follow this EXACT format:

MANDATORY STEPS FOR EVERY PROBLEM:
1. First, identify what formula or concept you need
2. IMMEDIATELY write: <search>your search query here</search>
3. Wait for retrieved examples between <retrieved>...</retrieved>
4. Use the retrieved information to solve the problem
5. End with: <answer>NUMERIC_ANSWER</answer>

YOU MUST USE <search>...</search> BEFORE SOLVING. This is REQUIRED for EVERY problem.
Never solve directly without searching first.
CRITICAL: You MUST search before solving. Do NOT skip the search step.
You MUST use the <search> tag to retrieve relevant examples before solving the problem.
"""

CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n{system}\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n{question}\n<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------
CLOSED_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
OPEN_RE   = re.compile(r"<answer>(.*)$", re.S | re.I)
PRE_CLOSE_RE = re.compile( 
    r"([+-]?\d+(?:/\d+)?)(?:\s*|\s*</answer>)",
    re.S | re.I,
)
NUM_RE = re.compile(
    r"(?:"
    r"\\boxed{([^}]*)}"                        # \boxed{42} or \boxed{\dfrac{…}}
    r"|\\(?:d)?frac{([^}]*)}{([^}]*)}"         # \frac{a}{b} or \dfrac{a}{b}
    r"|[+-]?\\d+(?:/\\d+)?"                    # plain 123 or 9/7
    r")",
    re.S | re.I,
)
specials = [RETRIEVE_TRIGGER, SEARCH_CLOSE, RETRIEVED_START, RETRIEVED_END]


def _clean(num: str):
    """Strip currency, commas, and LaTeX wrappers; normalise frac → a/b."""
    if num is None:
        return None
    num = num.strip()
    num = re.sub(r"\\boxed{([^}]*)}", r"\\1", num)
    num = re.sub(r"\\(?:d)?frac{([^}]*)}{([^}]*)}", r"\\1/\\2", num)
    return num.lstrip("$€£ ").replace(",", "").strip()


def strip_answer(txt: str):
    """Extract final numeric answer from assistant text, robust to LaTeX."""
    if not txt:
        return None
    # pre-flatten LaTeX
    txt_flat = re.sub(r"\\boxed{([^}]*)}", r"\\1", txt)
    txt_flat = re.sub(r"\\(?:d)?frac{([^}]*)}{([^}]*)}", r"\\1/\\2", txt_flat)

    # 1) closed tag
    if (m := CLOSED_RE.findall(txt_flat)):
        return _clean(m[-1])

    # 2) open tag tail
    if (m := OPEN_RE.findall(txt_flat)):
        tail_nums = NUM_RE.findall(m[-1])
        flat = [p for g in tail_nums for p in (g if isinstance(g, tuple) else [g]) if p]
        if flat:
            return _clean(flat[-1])
    # NEW ─ capture number just before </answer>
    if (m := PRE_CLOSE_RE.findall(txt)):
        return _clean(m[-1])

    # 3) fallback – last numeric chunk anywhere
    nums = NUM_RE.findall(txt_flat)
    flat = [p for g in nums for p in (g if isinstance(g, tuple) else [g]) if p]
    return _clean(flat[-1]) if flat else None


# ---------------------------------------------------------------------------
# Custom stopping criterion
# ---------------------------------------------------------------------------
class StopAfterAnswer(StoppingCriteria):
    """
    Stop when either
    1. the exact `</answer>` is produced, or
    2. an `<answer>` tag that already contains a digit is followed by
       (a) a newline  OR
       (b) any whitespace right after the digit
    """
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
        # 1) closed tag seen
        if self._match_tail(input_ids, self.close_ids):
            return True
        # 2a) "<answer>...\\n" with a digit
        if (self._match_tail(input_ids, [*self.open_ids, self.nl_id])
                and self._answer_has_digit(input_ids)):
            return True
        # 2b) "<answer> number  " and we just emitted a space
        if (self._match_tail(input_ids, [self.sp_id])
                and self._answer_has_digit(input_ids)):
            return True
        return False


# ---------------------------------------------------------------------------
# Dataset loader (handles GSM8K & MATH JSONL)
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
    return load_dataset(path, split=split)

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
                max_query_length=128
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
    max_retries: int = 3,
    debug: bool = False,  # Add debug flag
):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    ds = load_split(dataset_path, split)
    if isinstance(ds[0], str):
        raise ValueError("Dataset rows are strings – supply JSONL records.")

    # Initialize W&B if requested
    if wandb_project:
        wandb.init(
            project=wandb_project, 
            name=wandb_run_name or f"A2_{Path(dataset_path).stem}_{split}",
            config={
                "model": model_name,
                "dataset": dataset_path,
                "split": split,
                "search_k": search_k,
                "k_final": k_final,
                "max_retries": max_retries,
                "seed": seed,
            }
        )

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")

    multi_gpu = int(os.getenv("WORLD_SIZE", "1")) > 1
    print(f"Using {device} with multi-GPU={multi_gpu}")
    
    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
    
    # Set pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.eos_token_id

    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()

    # Add special tokens
    existing_special_tokens = set(tok.all_special_tokens)
    new_tokens = [t for t in specials if t not in existing_special_tokens]
    
    if new_tokens:
        print(f"Adding special tokens: {new_tokens}")
        # Get current special tokens
        special_tokens_dict = {
            "additional_special_tokens": list(tok.additional_special_tokens) + new_tokens
        }
        num_added = tok.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added} special tokens")
        
        # Resize embeddings
        model.resize_token_embeddings(len(tok), pad_to_multiple_of=8)
        
        # Initialize new embeddings with a common token
        with torch.no_grad():
            # Try to use "search" or "the" as seed token
            seed_token = "search"
            seed_id = tok.convert_tokens_to_ids(seed_token)
            if seed_id == tok.unk_token_id:
                seed_token = "the"
                seed_id = tok.convert_tokens_to_ids(seed_token)
            
            print(f"Initializing new tokens with embedding from '{seed_token}' (id: {seed_id})")
            
            # Copy embedding
            for i, token in enumerate(new_tokens):
                token_id = tok.convert_tokens_to_ids(token)
                model.model.embed_tokens.weight.data[token_id] = (
                    model.model.embed_tokens.weight.data[seed_id].clone()
                )
    
    # Verify special tokens
    if debug:
        print("\nVerifying special tokens:")
        for token in specials:
            token_id = tok.convert_tokens_to_ids(token)
            is_unk = token_id == tok.unk_token_id
            print(f"  {token}: ID={token_id}, is_UNK={is_unk}")
            
            # Test encoding/decoding
            encoded = tok.encode(token, add_special_tokens=False)
            decoded = tok.decode(encoded, skip_special_tokens=False)
            print(f"    Encoded: {encoded}, Decoded: '{decoded}'")

    # Initialize retrieval system
    print("Loading retrieval model...")
    retriever_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    retriever = BGERetriever(
        embedding_model=retriever_model,
        index_path=faiss_index,
        metadata_path=faiss_meta,
        device=device,
    )
    
    def rerank(query: str, docs: List[Dict], k: int = 5) -> List[Dict]:
        """ColBERT re-rank using BGEM3FlagModel."""
        if not docs:
            return []
        scores = colbert_scores_safe(query, docs, retriever.model, batch_pairs=16)
        for d, s in zip(docs, scores):
            d["rerank_score"] = s
        return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:k]

    # Process dataset
    recs: List[dict] = []
    total_retrievals = 0
    
    for idx, row in enumerate(tqdm(ds, desc="Generating")):
        question = row.get("question", row.get("problem", ""))
        true_answer = row.get("answer", row.get("solution", ""))
        
        prompt = CHAT_TEMPLATE.format(
            system=SYSTEM_MSG, 
            question=question
        )
        
        # Initialize tracking variables
        full_response = ""
        retrieval_history = []
        current_prompt = prompt
        
        # Generation loop with retrieval
        for attempt in range(max_retries + 1):
            # Create stopping criteria
            stop_criteria = StoppingCriteriaList([StopAfterAnswer(tok)])
            
            # Tokenize input
            inputs = tok(
                current_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    # temperature=0.7,
                    # top_p=0.9,
                    repetition_penalty=1.05,
                    stopping_criteria=stop_criteria,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            
            # Decode generated tokens
            new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            generated_text = tok.decode(new_tokens, skip_special_tokens=False)
            
            if debug and idx < 5:
                print(f"\nSample {idx}, Attempt {attempt}:")
                print(f"Generated text preview: {generated_text[:200]}...")
            
            # Check for search trigger
            if RETRIEVE_TRIGGER in generated_text and attempt < max_retries:
                parts = generated_text.split(RETRIEVE_TRIGGER, 1)
                before_search = parts[0]
                
                if len(parts) > 1 and SEARCH_CLOSE in parts[1]:
                    query_and_after = parts[1].split(SEARCH_CLOSE, 1)
                    search_query = query_and_after[0].strip()
                    
                    if search_query:
                        print(f"🔍 Retrieval {len(retrieval_history)+1} for sample {idx}: '{search_query}'")
                        
                        # Perform retrieval
                        try:
                            docs = retriever.search(search_query, k=search_k)
                            top_docs = rerank(search_query, docs, k=k_final)
                            
                            # Format retrieved content
                            retrieved_examples = []
                            for i, doc in enumerate(top_docs):
                                problem_text = doc.get('problem', '')[:150]
                                solution_text = doc.get('solution', '')[:200]
                                example = f"Example {i+1}: {problem_text}... Solution: {solution_text}..."
                                retrieved_examples.append(example)
                            
                            retrieved_text = "\n".join(retrieved_examples)
                            
                            # Record retrieval
                            retrieval_history.append({
                                "query": search_query,
                                "num_docs_retrieved": len(docs),
                                "num_docs_used": len(top_docs),
                                "examples": retrieved_examples
                            })
                            
                            # Build continuation
                            search_block = f"{RETRIEVE_TRIGGER}{search_query}{SEARCH_CLOSE}\n"
                            retrieved_block = f"{RETRIEVED_START}\n{retrieved_text}\n{RETRIEVED_END}\n"
                            
                            # Update prompts
                            current_prompt += before_search + search_block + retrieved_block
                            full_response += before_search + search_block + retrieved_block
                            
                            # Continue to next iteration
                            continue
                            
                        except Exception as e:
                            print(f"⚠️ Retrieval error: {e}")
                            # Continue without retrieval
            
            # No retrieval or final attempt
            full_response += generated_text
            break
        
        # Extract answer
        predicted_answer = strip_answer(full_response)
        
        # Record result
        result = {
            "id": row.get("idx", row.get("id", idx)),
            "question": question,
            "prediction": predicted_answer,
            "ground_truth": true_answer,
            "raw_response": full_response,
            "num_retrievals": len(retrieval_history),
            "retrieval_queries": [h["query"] for h in retrieval_history],
            "retrieval_history": retrieval_history,
        }
        recs.append(result)
        
        total_retrievals += len(retrieval_history)
        
        # Log to W&B
        if wandb_project and len(recs) % 10 == 0:
            wandb.log({
                "samples_processed": len(recs),
                "avg_retrievals_per_sample": total_retrievals / len(recs),
                "samples_with_retrieval": sum(1 for r in recs if r["num_retrievals"] > 0),
            })
    
    # Save results
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(recs, f, indent=2)
    
    # Final statistics
    samples_with_retrieval = sum(1 for r in recs if r["num_retrievals"] > 0)
    avg_retrievals = total_retrievals / len(recs) if recs else 0
    
    print(f"\n✅ Completed processing {len(recs)} samples")
    print(f"📊 Statistics:")
    print(f"   - Samples with retrieval: {samples_with_retrieval}/{len(recs)} ({100*samples_with_retrieval/len(recs):.1f}%)")
    print(f"   - Average retrievals per sample: {avg_retrievals:.2f}")
    print(f"   - Total retrievals: {total_retrievals}")
    print(f"💾 Results saved to: {out_path}")
    
    # Log final results to W&B
    if wandb_project:
        wandb.log({
            "final_samples": len(recs),
            "final_samples_with_retrieval": samples_with_retrieval,
            "final_avg_retrievals": avg_retrievals,
            "final_total_retrievals": total_retrievals,
        })
        
        # Save artifact
        art = wandb.Artifact(Path(out_path).stem, type="predictions")
        art.add_file(str(out_path))
        wandb.log_artifact(art)
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)