import os
import torch
from torch.utils.data import Dataset
import json

def load_gsm8k_from_file(path, limit=None):
    """Loads data from a GSM8K JSONL file, returning a list of dicts."""
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data_list.append(json.loads(line))
    return data_list

def load_math_from_dir(base_dir, subjects, limit=None):
    """Loads data from the Hendrycks MATH dataset directories."""
    data_list = []
    for subject in subjects:
        if limit and len(data_list) >= limit:
            break
        file_path = os.path.join(base_dir, subject, "test.jsonl")
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and len(data_list) >= limit:
                    break
                data_list.append(json.loads(line))
    return data_list

def collate_fn(batch_items):
    """Collates a batch of items into a dictionary with lists for each key."""
    return {key: [d[key] for d in batch_items] for key in batch_items[0]}

class MathDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_q_len, max_a_len,
                    question_key='problem', answer_key='solution'):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.question_key = question_key
        self.answer_key = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample.get(self.question_key, '')
        answer = sample.get(self.answer_key, '')
        question_encoding = self.tokenizer(question, max_length=self.max_q_len, padding='max_length', truncation=True, return_tensors='pt')
        labels = self.tokenizer(text_target=answer, max_length=self.max_a_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        return {
            'input_ids': question_encoding['input_ids'].squeeze(0),
            'attention_mask': question_encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'original_question': question,
            'original_answer': answer
        }