import json
import torch
from typing import List, Dict, Any

def load_local_nq_json(file_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    Loads NQ data from a local JSON file.
    Assumes the JSON file has a root key "data" containing a list of entries.

    Args:
        file_path: Path to the JSON file.
        limit: Optional maximum number of entries to load.

    Returns:
        A list of dictionaries, where each dictionary is an NQ entry.
    """
    print(f"Loading local NQ data from: {file_path}")
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
                data_list = content["data"]
            elif isinstance(content, list): # Fallback if the root is directly a list
                print("Warning: JSON file appears to be a root list, not a dict with a 'data' key.")
                data_list = content
            else:
                raise ValueError(f"JSON file {file_path} does not contain a 'data' key with a list of entries, or is not a root list of entries.")
        
        if limit is not None and limit > 0 and len(data_list) > limit:
            print(f"Limiting dataset from {len(data_list)} to first {limit} entries.")
            return data_list[:limit]
        elif limit is not None and limit > 0 :
             print(f"Dataset has {len(data_list)} entries, (requested limit: {limit}).")
        return data_list
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        raise
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        raise

def custom_collate_fn(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle batches of dictionaries from NQDataset.
    It stacks tensor items and collects string items into lists.
    """
    if not batch_items:
        return {}

    collated_batch = {}
    # Get keys from the first item, assuming all items have the same structure
    item_keys = batch_items[0].keys()

    for key in item_keys:
        # Collect all values for the current key from the batch
        values = [item[key] for item in batch_items]
        
        if isinstance(values[0], torch.Tensor):
            # If the item is a tensor, stack them
            collated_batch[key] = torch.stack(values)
        elif isinstance(values[0], str):
            # If the item is a string, keep it as a list of strings
            collated_batch[key] = values
        elif key == "precomputed_sparse_docs": # Example specific handling
            collated_batch[key] = values
        else:
            # For other types, just collect them in a list (e.g., int, float)
            # Or handle specifically if needed
            collated_batch[key] = values 
            
    return collated_batch

def load_precomputed_sparse_results(jsonl_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads pre-computed sparse retrieval results from a JSONL file into a dictionary.
    Assumes each line in the JSONL file is a JSON object with at least
    "example_id" (or "original_question") and "sparse_retrieved_docs".
    """
    lookup_dict = {}
    print(f"Loading pre-computed sparse results from: {jsonl_file_path} ...")
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    # Use example_id as the key if available and preferred
                    key = entry.get("example_id") 
                    if key is None: # Fallback to original_question if example_id is missing
                        key = entry.get("original_question")
                    
                    if key:
                        lookup_dict[key] = entry.get("sparse_retrieved_docs", [])
                    else:
                        print(f"Warning: Missing 'example_id' or 'original_question' key in line {line_num+1} of {jsonl_file_path}")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {line_num+1} in {jsonl_file_path}")
        print(f"Loaded {len(lookup_dict)} pre-computed sparse entries from {jsonl_file_path}.")
    except FileNotFoundError:
        print(f"Warning: Pre-computed sparse results file not found: {jsonl_file_path}. Sparse results will be empty.")
    except Exception as e:
        print(f"Error loading pre-computed sparse results from {jsonl_file_path}: {e}")
    return lookup_dict
    
def extract_final_numeric_answer(text):
    text = str(text).replace(',', '')
    tokens = re.findall(r"-?\d+\.\d+|-?\d+", text)
    return tokens[-1] if tokens else ""

def extract_gsm8k_gold_answer(text):
    if isinstance(text, str):
        m = re.search(r"####\s*([-\d\.,]+)", text)
        if m: return m.group(1).replace(',', '')
        tokens = re.findall(r"-?\d+\.\d+|-?\d+", text.replace(',', ''))
        return tokens[-1] if tokens else ""
    return str(text)

def compute_accuracy(preds, golds):
    correct = sum(1 for p, g in zip(preds, golds) if p.strip() == g.strip())
    return (correct / len(preds)) * 100 if preds else 0.0