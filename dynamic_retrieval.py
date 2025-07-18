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

SYSTEM_MSG = """
You are an expert competition mathematician with access to a retrieval tool.

Follow this STRICT protocol **zero-shot** (no examples given):

1. Begin every problem with a PLAN block listing each required formula, theorem, identity, definition, or fact as a bullet. For each item end the line with either KNOWN (you are fully certain) or UNKNOWN (any doubt).
Format:
<plan>
- Binomial theorem expansion: KNOWN
- Inclusion–Exclusion count of onto functions: UNKNOWN
</plan>

2. If there is at least one UNKNOWN item you MUST immediately output EXACTLY ONE search query on a single line inside <search>...</search> and then STOP (do not start solving yet).

3. After you receive a <retrieved>...</retrieved> block, update any remaining UNKNOWN items (mark them KNOWN if resolved) and if any still UNKNOWN, issue another <search>...</search> (one query) and STOP again.

4. Only when all items are KNOWN, write the detailed solution reasoning and finish with <answer>NUMERIC_ANSWER</answer>.

Rules:
- Never fabricate formulas; mark them UNKNOWN instead.
- Keep each search query concise (≤ 12 words, no punctuation except parentheses if essential).
- At most one query per <search> tag.
- If everything is simple arithmetic / direct recall, you may have an empty UNKNOWN set and skip searching.

If you are uncertain whether a formula name is exact, treat it as UNKNOWN.
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

# ---------------------------------------------------------------------------
# PLAN/UNKNOWN/auto-retrieval helpers
# ---------------------------------------------------------------------------
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.S | re.I)
UNKNOWN_LINE_RE = re.compile(r"\bUNKNOWN\b", re.I)

STOPWORDS = {
    "the","a","an","of","for","to","and","or","in","on","with","by",
    "from","using","count","number","find","compute","formula","identity",
    "function","value","values","prove","show","determine"
}

def extract_unknown_items(plan_text: str) -> list[str]:
    lines = []
    for raw in plan_text.splitlines():
        if UNKNOWN_LINE_RE.search(raw):
            # remove bullet markers
            line = re.sub(r"^[\s*-]+", "", raw).strip()
            # remove trailing 'UNKNOWN'
            line = re.sub(r"\bUNKNOWN\b", "", line, flags=re.I).strip(" :")
            if line:
                lines.append(line)
    return lines

def build_query_from_unknowns(unknowns: list[str], max_words: int = 12) -> str:
    # naive keyword extraction: take last 6 content words from each unknown line
    words = []
    for u in unknowns:
        tokens = re.findall(r"[A-Za-z]+", u.lower())
        content = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
        if not content:
            content = tokens[-3:]
        words.extend(content[-6:])
    # de-duplicate preserving order
    seen = set()
    dedup = []
    for w in words:
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    return " ".join(dedup[:max_words]) or "math problem formula"


# ---------------------------------------------------------------------------
# Reasoning enforcement helpers
# ---------------------------------------------------------------------------
MIN_REASON_TOKENS = 60          # minimum approximate tokens (words) between last </plan> and <answer>
MAX_EXPANSION_ROUNDS = 2        # safeguard to avoid infinite loops

ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.S | re.I)

def _reasoning_token_count(txt: str) -> int:
    """Approximate 'reasoning' length: words between last </plan> and first <answer>."""
    plan_close = txt.rfind("</plan>")
    if plan_close == -1:
        return 0
    ans_match = re.search(r"<answer>", txt, re.I)
    if not ans_match:
        return 0
    segment = txt[plan_close + len("</plan>"): ans_match.start()]
    # Strip retrieval blocks
    segment = re.sub(r"<retrieved>.*?</retrieved>", "", segment, flags=re.S | re.I)
    words = re.findall(r"\w+", segment)
    return len(words)

def _strip_last_answer(txt: str) -> str:
    """Remove the last <answer>...</answer> block (if any)."""
    matches = list(ANSWER_TAG_RE.finditer(txt))
    if not matches:
        return txt
    m = matches[-1]
    return txt[:m.start()]  # drop answer so we can force expansion

# ---------------------------------------------------------------------------
# Additional reasoning / answer control helpers
# ---------------------------------------------------------------------------
MIN_REASON_TOKENS_AFTER_RETR = 50   # reasoning tokens required AFTER last retrieval before answer allowed
COMPLEX_KEYWORDS = {
    "integral","probability","combinator","onto","surjective","inequality",
    "sequence","limit","geometry","fraction","ceiling","floor","mod","divisor",
    "prime","quadratic","polynomial","interest","compound","asymptote"
}

def _reasoning_token_count_after_last_retr(txt: str) -> int:
    """Words between last </retrieved> (or </plan> if none) and first <answer>."""
    base = txt.rfind(RETRIEVED_END)
    if base == -1:
        base = txt.rfind("</plan>")
    if base == -1:
        base = 0
    ans_match = re.search(r"<answer>", txt, re.I)
    if not ans_match:
        return 0
    segment = txt[base: ans_match.start()]
    segment = re.sub(r"<retrieved>.*?</retrieved>", "", segment, flags=re.S | re.I)
    words = re.findall(r"\w+", segment)
    return len(words)

def _is_complex(question: str) -> bool:
    qlow = question.lower()
    return any(k in qlow for k in COMPLEX_KEYWORDS) or len(qlow.split()) >= 18


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
    debug: bool = False,
    auto_plan_retrieval: bool = True,
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
    total_auto = 0
    
    for idx, row in enumerate(
        track(
            ds,
            description="Generating",
            total=len(ds),
            refresh_per_second=1
            )
        ): 
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
        answer_emitted = False
        last_retrieval_round = -1
        
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
                    stopping_criteria=stop_criteria,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            # Decode generated tokens
            new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            generated_text = tok.decode(new_tokens, skip_special_tokens=False)

            # Sanitize runaway plain 'search' repetitions (model sometimes loops)
            generated_text = re.sub(r'(?:\bsearch\b\s*){5,}', '', generated_text, flags=re.I)

            if debug and idx < 5:
                print(f"\nSample {idx}, Attempt {attempt}:")
                print(f"Generated text preview: {generated_text[:200]}...")

            # Zero-shot PLAN-based automatic retrieval (if model forgot to emit <search>)
            if auto_plan_retrieval and not retrieval_history and RETRIEVE_TRIGGER not in generated_text:
                # Accumulate current draft (plan likely at start of generated_text)
                draft_text = full_response + generated_text
                m = PLAN_RE.search(draft_text)
                if m:
                    plan_block = m.group(1)
                    unknowns = extract_unknown_items(plan_block)
                    if unknowns:
                        auto_query = build_query_from_unknowns(unknowns)
                        print(f"🔍 Auto retrieval (PLAN UNKNOWN) for sample {idx}: '{auto_query}'")
                        try:
                            docs = retriever.search(auto_query, k=search_k)
                            top_docs = rerank(auto_query, docs, k=k_final)
                            retrieved_examples = []
                            for i, doc in enumerate(top_docs):
                                problem_text = doc.get('problem', '')
                                solution_text = doc.get('solution', '')
                                example = f"Example {i+1}: {problem_text}... Solution: {solution_text}..."
                                retrieved_examples.append(example)
                            retrieved_text = "\n".join(retrieved_examples)
                            retrieval_history.append({
                                "query": auto_query,
                                "num_docs_retrieved": len(docs),
                                "num_docs_used": len(top_docs),
                                "examples": retrieved_examples,
                                "auto": True,
                            })
                            last_retrieval_round = len(retrieval_history)
                            search_block = f"{RETRIEVE_TRIGGER}{auto_query}{SEARCH_CLOSE}\n"
                            retrieved_block = f"{RETRIEVED_START}\n{retrieved_text}\n{RETRIEVED_END}\n"
                            current_prompt += generated_text + search_block + retrieved_block
                            full_response += generated_text + search_block + retrieved_block
                            continue  # go to next attempt
                        except Exception as e:
                            print(f"⚠️ Auto plan retrieval error: {e}")

            # Hard truncate after first closing </answer> to prevent trailing <search> or noise
            if "</answer>" in generated_text.lower():
                pre, _post = re.split(r"</answer>", generated_text, 1, flags=re.I)
                generated_text = pre + "</answer>"
                answer_emitted = True

            # Check for search trigger
            if (RETRIEVE_TRIGGER in generated_text) and (not answer_emitted) and attempt < max_retries:
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
                                problem_text = doc.get('problem', '')
                                solution_text = doc.get('solution', '')
                                example = f"Example {i+1}: {problem_text}... Solution: {solution_text}..."
                                retrieved_examples.append(example)

                            retrieved_text = "\n".join(retrieved_examples)

                            # Record retrieval
                            retrieval_history.append({
                                "query": search_query,
                                "num_docs_retrieved": len(docs),
                                "num_docs_used": len(top_docs),
                                "examples": retrieved_examples,
                                "auto": False,
                            })
                            last_retrieval_round = len(retrieval_history)

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

        # Enforce sufficient reasoning length if answer appeared too early
        expansion_round = 0
        while True:
            predicted_answer = strip_answer(full_response)
            reason_len = _reasoning_token_count(full_response)
            reason_after_last = _reasoning_token_count_after_last_retr(full_response)
            need_expansion = (
                predicted_answer is not None
                and (
                    (len(retrieval_history) > 0 and reason_after_last < MIN_REASON_TOKENS_AFTER_RETR)
                    or (len(retrieval_history) == 0 and _is_complex(question) and reason_len < MIN_REASON_TOKENS)
                )
                and expansion_round < MAX_EXPANSION_ROUNDS
            )
            if not need_expansion:
                break

            if debug and idx < 5:
                print(f"⚠️ Reasoning too short (total={reason_len}, post_retr={reason_after_last}); forcing expansion round {expansion_round+1}")

            # Remove the premature answer
            full_response = _strip_last_answer(full_response)

            # Append explicit expansion instruction
            expansion_instruction = (
                "\n<reasoning_expand>\n"
                "Your prior reasoning was too brief. Provide a detailed, step-by-step derivation, "
                "justifying each transformation. Do NOT output <answer> until the reasoning is complete.\n"
                "</reasoning_expand>\n"
            )
            current_prompt = full_response + expansion_instruction

            # Regenerate continuation (single attempt, no retrieval trigger here)
            stop_criteria = StoppingCriteriaList([StopAfterAnswer(tok)])
            inputs = tok(
                current_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    stopping_criteria=stop_criteria,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            continuation = tok.decode(new_tokens, skip_special_tokens=False)
            continuation = re.sub(r'(?:\bsearch\b\s*){5,}', '', continuation, flags=re.I)
            full_response += continuation
            expansion_round += 1

        # Extract final answer after any expansions
        predicted_answer = strip_answer(full_response)
        if predicted_answer is not None:
            answer_emitted = True
        
        # Record result
        result = {
            "id": row.get("idx", row.get("id", idx)),
            "question": question,
            "prediction": predicted_answer,
            "ground_truth": true_answer,
            "raw_response": full_response,
            "num_retrievals": len(retrieval_history),
            "num_auto_retrievals": sum(1 for h in retrieval_history if h.get("auto")),
            "reason_tokens_total": _reasoning_token_count(full_response),
            "reason_tokens_after_last_retr": _reasoning_token_count_after_last_retr(full_response),
            "retrieval_queries": [h["query"] for h in retrieval_history],
            "retrieval_history": retrieval_history,
        }
        recs.append(result)

        total_retrievals += len(retrieval_history)
        total_auto += sum(1 for h in retrieval_history if h.get("auto"))

        # Log to W&B
        if wandb_project and len(recs) % 10 == 0:
            wandb.log({
                "samples_processed": len(recs),
                "avg_retrievals_per_sample": total_retrievals / len(recs),
                "samples_with_retrieval": sum(1 for r in recs if r["num_retrievals"] > 0),
                "auto_retrieval_rate": total_auto / len(recs),
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
            "final_total_auto_retrievals": total_auto,
            "final_auto_retrieval_rate": (total_auto / len(recs)) if recs else 0,
        })
        
        # Save artifact
        art = wandb.Artifact(Path(out_path).stem, type="predictions")
        art.add_file(str(out_path))
        wandb.log_artifact(art)
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)