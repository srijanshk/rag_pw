import json
import nltk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# --- Configuration ---
INPUT_JSONL_FILE = "./thesis_datasets/openmathinstruct2/openmathinstruct2_train_streamed.jsonl" # Adjust if your path/filename is different
OUTPUT_DIR = "./thesis_datasets/openmathinstruct2_analysis" # Directory to save plots

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_solution_lengths():
    char_lengths = []
    word_lengths = []
    sentence_lengths = []

    print(f"Analyzing solution lengths from: {INPUT_JSONL_FILE}")

    # First pass to count total lines for tqdm
    total_lines = 0
    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as infile:
            for _ in infile:
                total_lines += 1
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_JSONL_FILE}")
        print("Please ensure you have downloaded and processed the OpenMathInstruct-2 dataset.")
        return

    if total_lines == 0:
        print("Input file is empty. No analysis to perform.")
        return

    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, total=total_lines, desc="Processing solutions"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # print(f"Warning: Skipping malformed JSON line.")
                continue

            solution_text = entry.get("generated_solution", "")
            if not solution_text:
                # Also count entries with no solution text if needed, or just skip
                char_lengths.append(0)
                word_lengths.append(0)
                sentence_lengths.append(0)
                continue

            # Character length
            char_lengths.append(len(solution_text))

            # Word length (simple split)
            word_lengths.append(len(solution_text.split()))

            # Sentence length (using NLTK)
            try:
                sentences = nltk.sent_tokenize(solution_text)
                sentence_lengths.append(len(sentences))
            except Exception as e:
                # print(f"Could not tokenize sentences for an entry: {e}")
                sentence_lengths.append(0) # Or handle differently

    if not char_lengths: # Should be redundant due to total_lines check, but good practice
        print("No solution texts found to analyze.")
        return

    # Convert to numpy arrays for easier stats
    char_lengths_np = np.array(char_lengths)
    word_lengths_np = np.array(word_lengths)
    sentence_lengths_np = np.array(sentence_lengths)

    # --- Print Statistics ---
    print("\n--- Solution Text Length Statistics ---")
    for name, data_np in [("Characters", char_lengths_np), 
                          ("Words", word_lengths_np), 
                          ("Sentences", sentence_lengths_np)]:
        print(f"\nDistribution by {name}:")
        print(f"  Min: {np.min(data_np):.0f}")
        print(f"  Max: {np.max(data_np):.0f}")
        print(f"  Mean: {np.mean(data_np):.2f}")
        print(f"  Median: {np.median(data_np):.0f}")
        print(f"  Std Dev: {np.std(data_np):.2f}")
        print(f"  Percentiles:")
        print(f"    25th: {np.percentile(data_np, 25):.0f}")
        print(f"    50th (Median): {np.percentile(data_np, 50):.0f}")
        print(f"    75th: {np.percentile(data_np, 75):.0f}")
        print(f"    90th: {np.percentile(data_np, 90):.0f}")
        print(f"    95th: {np.percentile(data_np, 95):.0f}")
        print(f"    99th: {np.percentile(data_np, 99):.0f}")

        # --- Plotting Histograms ---
        plt.figure(figsize=(10, 6))
        # Adjust bins based on data characteristics, exclude extreme outliers from default view for readability
        # For example, plot up to the 99th percentile to avoid very long tails skewing the main view
        upper_limit = np.percentile(data_np, 99.5) if len(data_np) > 0 else (1000 if name == "Characters" else 100)
        
        # Filter data for the main plot range to make it more readable
        plot_data = data_np[data_np <= upper_limit]
        if len(plot_data) == 0 and len(data_np) > 0: # if all data is above 99.5th (unlikely but handles edge case)
            plot_data = data_np

        num_bins = 50 if len(plot_data) > 50 else len(plot_data) # Avoid too many bins for small data
        if num_bins == 0 and len(data_np) > 0: # if plot_data is empty but data_np is not
             plt.hist(data_np, bins=50, color='skyblue', edgecolor='black')
        elif num_bins > 0 :
            plt.hist(plot_data, bins=num_bins, color='skyblue', edgecolor='black')
        
        plt.title(f"Distribution of Solution Length by {name} (up to {upper_limit:.0f} {name.lower()})")
        plt.xlabel(f"Number of {name}")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        
        plot_filename = os.path.join(OUTPUT_DIR, f"solution_length_distribution_{name.lower()}.png")
        plt.savefig(plot_filename)
        print(f"  Histogram saved to: {plot_filename}")
        plt.close()

if __name__ == "__main__":
    # Ensure NLTK's 'punkt' is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' tokenizer found.")
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Attempting to download...")
        try:
            nltk.download('punkt')
            print("'punkt' downloaded successfully.")
        except Exception as e:
            print(f"Failed to download 'punkt': {e}. Please do it manually: import nltk; nltk.download('punkt')")
            print("Exiting script.")
            exit()
    except ImportError:
        print("NLTK library not found. Please install it (`pip install nltk`) and download 'punkt'.")
        print("Exiting script.")
        exit()

    analyze_solution_lengths()