# import json

# path = "downloads/data/retriever/nq-train.json"

# with open(path, "r") as f:
#     data = json.load(f)

# for i in range(1):
#     print(f"📄 Entry {i+1}:")
#     print(data[i])
#     print("-" * 50)


# # import csv

# # tsv_path = "downloads/data/wikipedia_split/psgs_w100.tsv" 

# # with open(tsv_path, "r", encoding="utf-8") as f:
# #     reader = csv.reader(f, delimiter="\t")
# #     for i, row in enumerate(reader):
# #         print(f"Row {i+1}: {row}")
# #         if i >= 20:  # Show only the first 5 rows
# #             break
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2").to("cuda")
# grab 512 example passages
sample_texts = ["passage: Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members"] * 512

# warm up
_ = model.encode(sample_texts, batch_size=512, device="cuda")

# timed run
start = time.time()
_ = model.encode(sample_texts, batch_size=512, device="cuda")
elapsed = time.time() - start

secs_per_passage = elapsed / 512
# count total passages in your TSV (minus header)
total_passages = sum(1 for _ in open("downloads/data/wikipedia_split/psgs_w100.tsv")) - 1

estimated_seconds = total_passages * secs_per_passage
estimated_hours   = estimated_seconds / 3600

print(f"≈ {secs_per_passage:.4f}s per passage")
print(f"≈ {estimated_hours:.1f} hours to embed all passages")
