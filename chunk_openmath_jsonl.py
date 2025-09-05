# chunk_openmath_jsonl.py

import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Settings
INPUT_PATH = "./data/openmathinstruct2/openmathinstruct2_train_streamed.jsonl"
OUTPUT_PATH = "./data/openmathinstruct2/openmath_chunks.jsonl"
MAX_TOKENS = 7680
OVERLAP = 512

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(input_ids), max_tokens - overlap):
        end = start + max_tokens
        chunk_ids = input_ids[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk)
        if end >= len(input_ids):
            break
    return chunks

def process_jsonl(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(tqdm(fin, desc="Chunking")):
            try:
                item = json.loads(line)
                text = item.get("problem", "").strip() + "\n\nSolution:\n" + item.get("generated_solution", "").strip()
                chunks = chunk_text(text)
                for j, chunk in enumerate(chunks):
                    fout.write(json.dumps({
                        "chunk_id": f"{i}_{j}",
                        "example_id": i,
                        "text": chunk
                    }) + "\n")
            except Exception as e:
                print(f"⚠️ Skipping line {i}: {e}")
    print(f"✅ Saved chunks to {output_file}")

if __name__ == "__main__":
    process_jsonl(INPUT_PATH, OUTPUT_PATH)