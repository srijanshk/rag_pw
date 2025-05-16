import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Paths
METADATA = "/local00/student/shakya/wikipedia_metadata.jsonl"
TOKENIZER = "best_bart_model"    # or your fine-tuned checkpoint path
MAX_LEN    = 512
OUTPUT_IDS = "/local00/student/shakya/passage_input_ids.dat"
OUTPUT_MASK= "/local00/student/shakya/passage_attention_mask.dat"
MAP_JSON   = "/local00/student/shakya/id2row.json"

# 1) Collect IDs and texts
ids, texts = [], []
with open(METADATA) as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        pid   = str(entry["id"])
        txt   = entry["text"]
        if txt and txt.strip():
            ids.append(pid)
            texts.append(txt)

# 2) Tokenize in batches
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
num_passages = len(texts)
ids_mem   = np.memmap(OUTPUT_IDS, mode="w+", dtype=np.int64, shape=(num_passages, MAX_LEN))
mask_mem  = np.memmap(OUTPUT_MASK,mode="w+", dtype=np.int64, shape=(num_passages, MAX_LEN))

for start in tqdm(range(0, num_passages, 1000), desc="Tokenizing passages", unit="passage_batch"):
    batch_ids   = ids[start : start+1000]
    batch_texts = texts[start: start+1000]
    tok = tokenizer(batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="np")
    ids_mem [ start: start+len(batch_texts) ] = tok["input_ids"]
    mask_mem[ start: start+len(batch_texts) ] = tok["attention_mask"]

# 3) Save ID→row map
with open(MAP_JSON, "w") as f:
    json.dump({pid: i for i, pid in enumerate(ids)}, f)
