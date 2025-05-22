from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast # For type hinting, or AutoTokenizer

class NQDataset(Dataset):
    def __init__(self, 
                 data_list: List[Dict[str, Any]],
                 sparse_retrieval_data: Dict[str, List[Dict[str, Any]]], # Pre computed sparse retrieval data
                 question_tokenizer: PreTrainedTokenizerFast,
                 generator_tokenizer: PreTrainedTokenizerFast,
                 max_question_length: int, 
                 max_answer_length: int):
        """
        Args:
            data_list: A list of dictionary objects, where each dict is an NQ entry.
            question_tokenizer: Tokenizer for the question encoder.
            generator_tokenizer: Tokenizer for the generator (for labels).
            max_question_length: Max length for tokenized questions.
            max_answer_length: Max length for tokenized answers (labels).
        """
        self.dataset_entries = data_list
        self.sparse_retrieval_data = sparse_retrieval_data
        self.question_tokenizer = question_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        print(f"NQDataset initialized with {len(self.dataset_entries)} entries.")

    def __len__(self):
        return len(self.dataset_entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.dataset_entries[idx]
        
        original_question_str = entry.get("question", "")
        
        short_answers_list = entry.get("short_answers", [])
        # Ensure short_answers_list contains strings, take the first one

        precomputed_sparse_docs = self.sparse_retrieval_data.get(original_question_str, [])
        original_answer_str = ""
        if short_answers_list:
            if isinstance(short_answers_list[0], str):
                original_answer_str = short_answers_list[0]
            # Add more sophisticated handling if short_answers can have other structures

        if not original_question_str:
            # print(f"Warning: Missing question for entry at index {idx}. Using empty string.")
            pass # Tokenizer will handle empty string

        # Tokenize question for the E5 question encoder
        question_tokenized = self.question_tokenizer(
            original_question_str,
            max_length=self.max_question_length,
            padding="max_length", # Pad to max_length
            truncation=True,
            return_tensors="pt" # Return PyTorch tensors
        )

        labels_tokenized = self.generator_tokenizer(
            text_target=original_answer_str, # Use text_target for labels
            max_length=self.max_answer_length,
            padding="max_length", # Pad to max_length
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": question_tokenized.input_ids.squeeze(0), # Remove batch dim
            "attention_mask": question_tokenized.attention_mask.squeeze(0), # Remove batch dim
            "labels": labels_tokenized.input_ids.squeeze(0), # Remove batch dim
            "original_question": original_question_str,
            "original_answer": original_answer_str,
            "precomputed_sparse_docs": precomputed_sparse_docs
        }