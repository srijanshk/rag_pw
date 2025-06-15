import json
import nltk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv 
import sys

INPUT_TSV_FILE = "./thesis_datasets/openmathinstruct2/openmath_chunked_for_indexing.tsv"
OUTPUT_DIR = "./thesis_datasets/openmathinstruct2_analysis"

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_solution_lengths():
    char_lengths = []
    word_lengths = []
    sentence_lengths = []

    print(f"Analyzing solution chunks from: {INPUT_TSV_FILE}")

    # First pass to count total lines for tqdm (-1 for header)
    total_lines = 0
    try:
        with open(INPUT_TSV_FILE, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for line in infile) - 1
            if total_lines < 0: total_lines = 0
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_TSV_FILE}")
        return

    if total_lines == 0:
        print("Input file is empty or only contains a header. No analysis to perform.")
        return

    with open(INPUT_TSV_FILE, 'r', encoding='utf-8') as infile:
        # Use DictReader to handle the tab-separated format and header row
        reader = csv.DictReader(infile, delimiter='\t')
        for row in tqdm(reader, total=total_lines, desc="Processing solution chunks"):
            # Target the 'solution_chunk' column from your TSV
            solution_text = row.get("solution_chunk", "")
            
            if not solution_text:
                char_lengths.append(0)
                word_lengths.append(0)
                sentence_lengths.append(0)
                continue

            # Character length
            char_lengths.append(len(solution_text))

            # Word length
            word_lengths.append(len(solution_text.split()))

            # Sentence length
            try:
                sentences = nltk.sent_tokenize(solution_text)
                sentence_lengths.append(len(sentences))
            except Exception:
                sentence_lengths.append(0)

    if not char_lengths:
        print("No solution chunks found to analyze.")
        return

    # --- The statistics and plotting logic remains the same, as it's excellent ---
    char_lengths_np = np.array(char_lengths)
    word_lengths_np = np.array(word_lengths)
    sentence_lengths_np = np.array(sentence_lengths)

    print("\n--- Solution Chunk Length Statistics ---")
    for name, data_np in [("Characters", char_lengths_np), 
                          ("Words", word_lengths_np), 
                          ("Sentences", sentence_lengths_np)]:
        print(f"\nDistribution by {name}:")
        print(f"  Min: {np.min(data_np):.0f}")
        print(f"  Max: {np.max(data_np):.0f}")
        print(f"  Mean: {np.mean(data_np):.2f}")
        print(f"  Median: {np.median(data_np):.0f}")
        print(f"  Std Dev: {np.std(data_np):.2f}")
        print(f"  95th Percentile: {np.percentile(data_np, 95):.0f}")
        print(f"  99th Percentile: {np.percentile(data_np, 99):.0f}")

        plt.figure(figsize=(10, 6))
        upper_limit = np.percentile(data_np, 99.5) if len(data_np) > 0 else 1000
        plot_data = data_np[data_np <= upper_limit]
        if len(plot_data) == 0: plot_data = data_np
        
        plt.hist(plot_data, bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Distribution of Solution Chunk Length by {name} (up to 99.5th percentile)")
        plt.xlabel(f"Number of {name}")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        
        plot_filename = os.path.join(OUTPUT_DIR, f"solution_chunk_length_dist_{name.lower()}.png")
        plt.savefig(plot_filename)
        print(f"  Histogram saved to: {plot_filename}")
        plt.close()

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
    
    analyze_solution_lengths()