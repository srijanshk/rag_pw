# MathDataset.py

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from typing import Any, Dict, List
import json

def load_math_data(path, limit=None):
    """
    Loads a math dataset (like GSM8K) from a JSONL file.
    Expects each line to have "question" and "answer" (containing the full solution).
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            question = entry.get("question") or entry.get("problem")
            solution = entry.get("answer") or entry.get("solution")
            if question and solution:
                data.append({"question": question, "solution": solution})
            if limit and len(data) >= limit:
                break
    return data

def load_gsm8k_data(path, limit=None):
    """Loads questions and full solutions from a GSM8K jsonl file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            entry = json.loads(line)
            data.append({
                "question": entry.get("question", ""),
                "solution": entry.get("answer", "")
            })
    return data

class MathDataset(Dataset):
    def __init__(self, 
                 data_list: List[Dict[str, Any]],
                 retriever_tokenizer: PreTrainedTokenizerFast,
                 generator_tokenizer: PreTrainedTokenizerFast,
                 max_question_length: int, 
                 max_solution_length: int):
        self.dataset_entries = data_list
        self.retriever_tokenizer = retriever_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.max_question_length = max_question_length
        self.max_solution_length = max_solution_length

    def __len__(self):
        return len(self.dataset_entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.dataset_entries[idx]
        
        question_str = entry.get("question", "")
        solution_str = entry.get("solution", "")

        # Tokenize question for the retriever (e.g., E5)
        question_tokenized = self.retriever_tokenizer(
            f"query: {question_str}", # Use e5-style prefix
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize the FULL SOLUTION for the generator's labels
        labels_tokenized = self.generator_tokenizer(
            text_target=solution_str,
            max_length=self.max_solution_length, # This may need to be larger now
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": question_tokenized.input_ids.squeeze(0),
            "attention_mask": question_tokenized.attention_mask.squeeze(0),
            "labels": labels_tokenized.input_ids.squeeze(0),
            "original_question": question_str,
            "original_solution": solution_str,
        }