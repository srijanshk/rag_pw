import json
import nltk # Ensure NLTK is imported
from tqdm import tqdm
import os # For path operations

# --- Configuration ---
# Ensure this points to the JSONL file you created from the streamed download
INPUT_JSONL_FILE = "./thesis_datasets/openmathinstruct2/openmathinstruct2_train_streamed.jsonl" 
OUTPUT_TSV_FILE = "./thesis_datasets/openmathinstruct2/openmath_chunked_for_indexing.tsv"

# CHUNK_METHOD is now implicitly "sentence" by using the nltk_chunk_text_by_sentences function directly
SENTENCES_PER_CHUNK = 3 # Example: Group 3 sentences into one chunk. Adjust as needed.

# --- NLTK Sentence Chunker ---
def nltk_chunk_text_by_sentences(text, sentences_per_chunk=3):
    """
    Chunks text into groups of N sentences using NLTK.
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        print("NLTK 'punkt' tokenizer model not found. Please download it by running: import nltk; nltk.download('punkt')")
        # As a fallback, or you could raise an error to stop the script
        print("Falling back to naive sentence splitting due to missing 'punkt'. This is not recommended.")
        sentences = text.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        print(f"An error occurred during NLTK sentence tokenization: {e}")
        print("Falling back to naive sentence splitting. This is not recommended.")
        sentences = text.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk_sentences = []
    for i, sentence in enumerate(sentences):
        current_chunk_sentences.append(sentence)
        if (i + 1) % sentences_per_chunk == 0 or (i + 1) == len(sentences):
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []
    return chunks

# --- Main Preprocessing Logic ---
def preprocess_and_chunk_to_tsv():
    print(f"Starting preprocessing of {INPUT_JSONL_FILE} using NLTK sentence chunking...")
    chunk_id_counter = 0 # Simple global counter for unique chunk IDs

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_TSV_FILE), exist_ok=True)

    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_TSV_FILE, 'w', encoding='utf-8') as outfile:
        
        # Write TSV header - including expected_answer and problem_source
        outfile.write("id\tproblem\tsolution\texpected_answer\tproblem_source\n")

        for line_num, line in enumerate(tqdm(infile, desc="Processing lines")):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {line_num + 1}")
                continue

            # Using field names from your screenshot
            original_problem_statement = entry.get("problem", "") 
            solution_text_to_chunk = entry.get("generated_solution", "") 
            expected_answer_val = entry.get("expected_answer", "")
            problem_source_val = entry.get("problem_source", "")
            
            base_entry_id_for_chunks = f"entry{line_num:06d}" 

            if not solution_text_to_chunk:
                # print(f"Warning: No solution text (generated_solution) for entry on line {line_num + 1}. Skipping.")
                continue
            if not original_problem_statement:
                 print(f"Warning: No problem statement for entry on line {line_num + 1}. Using empty title.")

            # Apply NLTK sentence chunking
            chunks = nltk_chunk_text_by_sentences(solution_text_to_chunk, SENTENCES_PER_CHUNK)

            for chunk_num, chunk_content in enumerate(chunks):
                if not chunk_content.strip(): # Skip empty chunks
                    continue
                
                chunk_id_counter += 1
                # Create a unique chunk ID. Using a global counter is simple,
                # but f"{base_entry_id_for_chunks}_chk{chunk_num:03d}" would be more traceable.
                unique_chunk_id = f"chunk_{chunk_id_counter:07d}" 
                
                # Escape tabs and newlines in content to ensure valid TSV
                safe_title = str(original_problem_statement).replace('\t', ' ').replace('\n', ' ')
                safe_chunk_content = str(chunk_content).replace('\t', ' ').replace('\n', ' ')
                safe_expected_answer = str(expected_answer_val).replace('\t', ' ').replace('\n', ' ')
                safe_problem_source = str(problem_source_val).replace('\t', ' ').replace('\n', ' ')

                outfile.write(f"{unique_chunk_id}\t{safe_title}\t{safe_chunk_content}\t{safe_expected_answer}\t{safe_problem_source}\n")
            
            if (line_num + 1) % 10000 == 0:
                 print(f"Processed {line_num + 1} original entries, created {chunk_id_counter} chunks...")

    print(f"Preprocessing complete. Output TSV: {OUTPUT_TSV_FILE}")
    print(f"Total chunks created: {chunk_id_counter}")

if __name__ == "__main__":
    # Check for NLTK 'punkt' resource before starting
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' tokenizer found.")
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Please download it first.")
        print("You can do this by running the following in a Python interpreter:")
        print("import nltk")
        print("nltk.download('punkt')")
        print("Exiting script. Please download 'punkt' and re-run.")
        exit() # Exit if 'punkt' is not available
    except ImportError:
        print("NLTK library not found. Please install it (`pip install nltk`) and download 'punkt'.")
        print("Exiting script.")
        exit() # Exit if NLTK itself is not available
        
    preprocess_and_chunk_to_tsv()