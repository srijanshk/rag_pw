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
        else:
            # For other types, just collect them in a list (e.g., int, float)
            # Or handle specifically if needed
            collated_batch[key] = values 
            
    return collated_batch
    
