import os
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
BASE_OUTPUT_DIR = "./thesis_datasets"  # Base directory to save datasets
GSM8K_PATH_NAME = "openai/gsm8k"
GSM8K_SUBSET = "main"
MATH_PATH_NAME = "EleutherAI/hendrycks_math"
OPENMATH_PATH_NAME = "nvidia/OpenMathInstruct-2"
OPENMATH_SPLIT = "train" # Or "train_5M" for a smaller subset, check dataset card

# --- Helper Functions ---
def ensure_dir(directory_path):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def save_dataset_to_jsonl(dataset, file_path):
    """Saves a Hugging Face dataset to a JSONL file."""
    print(f"Saving dataset to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in tqdm(dataset, desc=f"Writing to {os.path.basename(file_path)}"):
            f.write(json.dumps(entry) + '\n')
    print(f"Successfully saved dataset to {file_path}")

# --- Main Download Logic ---
def download_benchmark_datasets():
    """Downloads GSM8K and MATH datasets."""
    print("--- Downloading Benchmark Datasets ---")

    # GSM8K
    gsm8k_output_dir = os.path.join(BASE_OUTPUT_DIR, "gsm8k")
    ensure_dir(gsm8k_output_dir)
    print(f"\nDownloading GSM8K ({GSM8K_PATH_NAME}, subset: {GSM8K_SUBSET})...")
    try:
        gsm8k_dataset_dict = load_dataset(GSM8K_PATH_NAME, GSM8K_SUBSET)
        for split_name, dataset in gsm8k_dataset_dict.items():
            output_file = os.path.join(gsm8k_output_dir, f"{split_name}.jsonl")
            save_dataset_to_jsonl(dataset, output_file)
        print(f"GSM8K dataset downloaded and saved to {gsm8k_output_dir}")
    except Exception as e:
        print(f"Error downloading/saving GSM8K: {e}")

    # MATH
    math_output_dir = os.path.join(BASE_OUTPUT_DIR, "math_hendrycks") # Changed dir name slightly for clarity
    ensure_dir(math_output_dir)
    print(f"\nDownloading MATH ({MATH_PATH_NAME})...")
    
    math_configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    
    for config_name in math_configs:
        print(f"  Downloading MATH subset: {config_name}...")
        try:
            # Load specific config (subset)
            math_dataset_dict = load_dataset(MATH_PATH_NAME, name=config_name, trust_remote_code=True)
            
            subset_output_dir = os.path.join(math_output_dir, config_name)
            ensure_dir(subset_output_dir)

            for split_name, dataset in math_dataset_dict.items():
                # Save each split (e.g., train, test) of the subset
                output_file = os.path.join(subset_output_dir, f"{split_name}.jsonl")
                save_dataset_to_jsonl(dataset, output_file)
            print(f"  MATH subset '{config_name}' downloaded and saved to {subset_output_dir}")
        except Exception as e:
            print(f"  Error downloading/saving MATH subset '{config_name}': {e}")
    print(f"All attempted MATH subsets processed. Check {math_output_dir}")



def download_and_stream_convert_openmathinstruct():
    """
    Downloads nvidia/OpenMathInstruct-2 using streaming
    and saves it directly to JSONL.
    """
    print("\n--- Streaming Download and Conversion for nvidia/OpenMathInstruct-2 ---")
    openmath_output_dir = os.path.join(BASE_OUTPUT_DIR, "openmathinstruct2")
    ensure_dir(openmath_output_dir)

    output_jsonl_file = os.path.join(openmath_output_dir, f"openmathinstruct2_{OPENMATH_SPLIT}_streamed.jsonl")

    if os.path.exists(output_jsonl_file):
        print(f"JSONL file already exists at {output_jsonl_file}. Skipping.")
        return

    print(f"Streaming {OPENMATH_PATH_NAME} (split: {OPENMATH_SPLIT}) and saving to {output_jsonl_file}...")
    
    try:
        # Load the dataset in streaming mode
        # Note: For streaming, individual files are still downloaded progressively.
        # The main benefit is avoiding loading the entire dataset into RAM.
        dataset_stream: IterableDataset = load_dataset(
            OPENMATH_PATH_NAME,
            split=OPENMATH_SPLIT,
            streaming=True
        )
        
        print(f"Opened stream for {OPENMATH_PATH_NAME} (split: {OPENMATH_SPLIT}). Writing to JSONL...")

        count = 0
        # It's good practice to try and get the total number of examples for tqdm if available
        # For some streamed datasets, this info might not be readily available or might be an estimate.
        total_examples = None
        try:
            # Attempt to get dataset info for the total number of examples
            ds_info = load_dataset(OPENMATH_PATH_NAME, name=OPENMATH_SPLIT, trust_remote_code=True) # Load non-streamed once just for info
            total_examples = ds_info[OPENMATH_SPLIT].num_rows if OPENMATH_SPLIT in ds_info else None
            print(f"Total examples to stream (approx.): {total_examples if total_examples else 'Unknown'}")
        except Exception as e:
            print(f"Could not get exact total number of examples beforehand: {e}")


        with open(output_jsonl_file, 'w', encoding='utf-8') as f:
            # Use tqdm for progress, it will update per iteration.
            # If total_examples is None, tqdm won't show a percentage completion but will show iteration count.
            for entry in tqdm(dataset_stream, desc=f"Streaming to {os.path.basename(output_jsonl_file)}", total=total_examples, unit=" examples"):
                f.write(json.dumps(entry) + '\n')
                count += 1
        
        print(f"Successfully streamed and saved {count} entries to {output_jsonl_file}")

    except Exception as e:
        print(f"Error streaming or saving nvidia/OpenMathInstruct-2: {e}")
        print("Please ensure you have 'pyarrow' installed if dealing with Parquet files (`pip install pyarrow`).")
        print("Check your internet connection and disk space.")

if __name__ == "__main__":
    ensure_dir(BASE_OUTPUT_DIR)
    # download_benchmark_datasets()
    download_and_stream_convert_openmathinstruct()
    print("\n--- All Dataset Download Tasks Attempted ---")
    print(f"Please check the '{BASE_OUTPUT_DIR}' directory for the downloaded datasets.")