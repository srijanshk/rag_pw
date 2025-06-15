import csv
import sys
from tqdm import tqdm
import os

# --- Configuration ---
# The path to your messy input file
INPUT_TSV_FILE = "./thesis_datasets/openmathinstruct2/openmath_chunked_for_indexing.tsv"

# The path where the new, clean file will be saved
OUTPUT_TSV_FILE = "./thesis_datasets/openmathinstruct2/openmath_chunked_for_indexing_CLEAN.tsv"

# --- Main Cleaning Logic ---

def sanitize_for_tsv(text: str) -> str:
    """
    Removes characters that interfere with TSV formatting.
    Replaces tabs and newlines with spaces.
    """
    if not isinstance(text, str):
        return ""
    # Replace tabs with a single space
    text = text.replace('\t', ' ')
    # Replace newlines with a single space
    text = text.replace('\n', ' ')
    # Optional: collapse multiple spaces into one
    text = ' '.join(text.split())
    return text

def main():
    print(f"Starting to clean file: {INPUT_TSV_FILE}")
    
    # Increase the field size limit to handle very large cells
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(1024 * 1024 * 1024) # Fallback to 1GB

    # Count total lines for a progress bar
    try:
        with open(INPUT_TSV_FILE, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_TSV_FILE}")
        return

    print(f"Found {total_lines} total lines to process...")

    # Open both files for reading and writing
    with open(INPUT_TSV_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_TSV_FILE, 'w', encoding='utf-8', newline='') as outfile:
        
        # Use csv reader/writer to correctly handle TSV format
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        # Process the header
        try:
            header = next(reader)
            writer.writerow(header)
            print(f"Header found and written to new file: {header}")
        except StopIteration:
            print("ERROR: Input file is empty.")
            return

        # Process the rest of the file with a progress bar
        processed_count = 0
        for row in tqdm(reader, total=total_lines - 1, desc="Cleaning rows"):
            # Sanitize each cell in the row
            sanitized_row = [sanitize_for_tsv(cell) for cell in row]
            
            # Ensure the row has the same number of columns as the header
            if len(sanitized_row) == len(header):
                writer.writerow(sanitized_row)
                processed_count += 1
            else:
                print(f"Warning: Skipping malformed row with {len(sanitized_row)} columns (expected {len(header)}). Content: {row}")

    print("\n" + "="*50)
    print("✅ Cleaning complete.")
    print(f"Successfully processed and wrote {processed_count} rows.")
    print(f"Your clean data is now available at: {OUTPUT_TSV_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()