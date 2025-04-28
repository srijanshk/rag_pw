import nltk
nltk.download('punkt_tab')

import os
import re
import json
import logging
from typing import Generator, List, Dict

import numpy as np
import faiss
from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import multiprocessing as mp

# Configuration
CONFIG = {
    "input_dir": "wiki_extracted",
    "output_dir": "processed_chunks",
    "index_path": "wikipedia_faiss_index",
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "max_token_length": 384,
    "target_chunk_tokens": 384,
    "min_chunk_tokens": 0,
    "overlap_sentences": 2,
    "batch_size": 16,
    "faiss_metric": faiss.METRIC_INNER_PRODUCT,
    "metadata_path": "merged_metadata.jsonl",
    "num_processes": 7,  # number of parallel processes
    "partial_embeddings_dir": "partial_embeddings",
}

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

import transformers
transformers.logging.set_verbosity_error()

def clean_wiki_text(text: str) -> str:
    """Clean Wikipedia text with aggressive regex rules"""
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)  # Templates
    text = re.sub(r"\[\[.*?\]\]", "", text)  # Links
    text = re.sub(r"==.*?==", "", text)  # Headers
    text = re.sub(r"\*+", "", text)  # Lists
    text = re.sub(r"#REDIRECT.*", "", text, flags=re.I)  # Redirects
    text = re.sub(r"\[\d+\]", "", text)  # Citations
    text = re.sub(r"\s+", " ", text)  # Whitespace
    text = re.sub(r"[^\w\s.,;:!?\-'\"()]", "", text)  # Special chars
    return text.strip()

class WikiChunker:
    """Semantic-aware Wikipedia chunker with token limits"""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    def chunk_article(self, text: str, title: str, url: str) -> Generator[Dict, None, None]:
        """Generate document chunks with metadata"""
        text = clean_wiki_text(text)
        if not text:
            return

        sentences = sent_tokenize(text)
        if not sentences:
            return

        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        while sentences:
            sentence = sentences.pop(0)
            sentence_tokens = len(self.tokenizer.tokenize(sentence))

            # If the sentence itself is too long, split it into smaller parts
            if sentence_tokens > CONFIG["max_token_length"]:
                tokens = self.tokenizer.tokenize(sentence)
                sub_token_lists = [
                    tokens[i: i + CONFIG["max_token_length"]]
                    for i in range(0, len(tokens), CONFIG["max_token_length"])
                ]
                sub_sentences = [
                    self.tokenizer.convert_tokens_to_string(sub_tokens)
                    for sub_tokens in sub_token_lists
                ]
                sentences = sub_sentences + sentences
                continue

            # Start new chunk if adding sentence exceeds max length
            if current_tokens + sentence_tokens > CONFIG["max_token_length"]:
                if current_chunk:
                    yield self._create_chunk(current_chunk, title, chunk_id, url)
                    chunk_id += 1
                    # Preserve overlap
                    current_chunk = current_chunk[-CONFIG["overlap_sentences"]:]
                    current_tokens = sum(len(self.tokenizer.tokenize(s)) for s in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

            # Commit chunk if reaches target length
            if current_tokens >= CONFIG["target_chunk_tokens"]:
                yield self._create_chunk(current_chunk, title, chunk_id, url)
                chunk_id += 1
                current_chunk = []
                current_tokens = 0

        # Handle remaining sentences
        if current_chunk and current_tokens >= CONFIG["min_chunk_tokens"]:
            yield self._create_chunk(current_chunk, title, chunk_id, url)

    def _create_chunk(self, sentences: List[str], title: str, chunk_id: int, url: str) -> Dict:
        """Format chunk with metadata"""
        text = " ".join(sentences).strip()
        return {
            "id": f"{title.replace(' ', '_')}_{chunk_id}",
            "chunk_id": chunk_id,
            "title": title,
            "text": text,
            "url": url,
            "token_count": len(self.tokenizer.tokenize(text)),
        }

def process_file(task):
    """
    Process a single file.
    Each task is a tuple: (input_path, output_path).
    """
    input_path, output_path = task
    document_count = 0
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        chunker = WikiChunker()
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                # Process the article into chunks
                chunks = chunker.chunk_article(data["text"], data["title"], data["url"])
                for chunk in chunks:
                    fout.write(json.dumps(chunk) + "\n")
                document_count += 1
        logger.info(f"Processed {document_count} documents from {input_path}.")
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
    return document_count

def process_wikipedia_parallel():
    """Main processing pipeline using multiprocessing."""
    tasks = []
    # Walk through the input directory and build tasks for each file.
    for root, dirs, files in os.walk(CONFIG["input_dir"]):
        relative_path = os.path.relpath(root, CONFIG["input_dir"])
        output_root = os.path.join(CONFIG["output_dir"], relative_path)
        os.makedirs(output_root, exist_ok=True)
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_root, file + ".jsonl")
            tasks.append((input_path, output_path))

    total_docs = 0
    with mp.Pool(CONFIG["num_processes"]) as pool:
        # Use imap_unordered for asynchronous processing of files.
        for count in tqdm(pool.imap_unordered(process_file, tasks),
                          total=len(tasks), desc="Processing files"):
            total_docs += count
    logger.info(f"Total documents processed: {total_docs}")

if __name__ == "__main__":
    process_wikipedia_parallel()
