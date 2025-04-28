import json

path = "downloads/data/retriever/nq-train.json"

with open(path, "r") as f:
    data = json.load(f)

for i in range(1):
    print(f"📄 Entry {i+1}:")
    print(data[i])
    print("-" * 50)


# import csv

# tsv_path = "downloads/data/wikipedia_split/psgs_w100.tsv" 

# with open(tsv_path, "r", encoding="utf-8") as f:
#     reader = csv.reader(f, delimiter="\t")
#     for i, row in enumerate(reader):
#         print(f"Row {i+1}: {row}")
#         if i >= 20:  # Show only the first 5 rows
#             break

