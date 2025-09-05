import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import random

CHUNK_PATH = "./data/openmathinstruct2/openmath_chunks.jsonl"
INDEX_PATH = "/local00/student/shakya/bge_m3_faiss.index"
ID_MAP_PATH = "/local00/student/shakya/bge_m3_chunk_ids.json"
EMBEDDINGS_SAVE_PATH = "/local00/student/shakya/bge_m3_embeddings.npy"
USE_IVFPQ = True  # enable this for scalable vector quantization
BATCH_SIZE = 50000  # batch size for adding vectors to FAISS
TRAIN_SAMPLE_SIZE = 100000  # sample size for training IVFPQ

def load_chunks(path):
    texts, ids = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            ids.append(obj["chunk_id"])
    return texts, ids

def encode_chunks_bge(texts, batch_size=32, max_length=8192):
    print("🔄 Loading BGE-M3 model with GPU support...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda", normalize_embeddings=True)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔍 Encoding embeddings"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, max_length=max_length)["dense_vecs"]
        embeddings.append(embs)
        # Save intermediate embeddings to disk to avoid memory issues
        if (i // batch_size + 1) % 100 == 0:
            combined = np.vstack(embeddings).astype("float32")
            np.save(EMBEDDINGS_SAVE_PATH, combined)
            print(f"💾 Saved intermediate embeddings up to index {i + batch_size}")
    combined = np.vstack(embeddings).astype("float32")
    np.save(EMBEDDINGS_SAVE_PATH, combined)
    print(f"💾 All embeddings saved to {EMBEDDINGS_SAVE_PATH}")
    return combined

def build_faiss_index(embeddings, index_path, ids_path, chunk_ids, nlist=256, m=32):
    dim = embeddings.shape[1]
    print(f"✅ Embedding dim = {dim}, num_vectors = {len(embeddings)}")

    if USE_IVFPQ:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
        train_sample_indices = random.sample(range(len(embeddings)), min(TRAIN_SAMPLE_SIZE, len(embeddings)))
        train_sample = embeddings[train_sample_indices]
        print(f"🔧 Training IVFPQ with {len(train_sample)} samples...")
        index.train(train_sample)
    else:
        index = faiss.IndexFlatL2(dim)

    if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        print(f"🚀 Using {faiss.get_num_gpus()} GPU(s) for FAISS")
        index = faiss.index_cpu_to_all_gpus(index)

    # Add vectors in batches to avoid memory overflow
    for i in tqdm(range(0, len(embeddings), BATCH_SIZE), desc="➕ Adding to index"):
        index.add(embeddings[i:i+BATCH_SIZE])
        print(f"   Added {min(i+BATCH_SIZE, len(embeddings))}/{len(embeddings)} vectors")

    print(f"✅ Total vectors in index: {index.ntotal}")

    # Save the FAISS index, falling back if GPU index cannot be converted
    if hasattr(faiss, "index_gpu_to_cpu"):
        try:
            faiss.write_index(faiss.index_gpu_to_cpu(index), index_path)
        except Exception:
            faiss.write_index(index, index_path)
    else:
        faiss.write_index(index, index_path)

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)
    print(f"💾 Index saved to {index_path}")
    print(f"💾 Chunk ID mapping saved to {ids_path}")

if __name__ == "__main__":
    texts, chunk_ids = load_chunks(CHUNK_PATH)
    embeddings = encode_chunks_bge(texts)
    build_faiss_index(embeddings, INDEX_PATH, ID_MAP_PATH, chunk_ids)