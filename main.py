import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs("checkpoints", exist_ok=True)

import json
import matplotlib.pyplot as plt
from Dense_Retriever import DenseRetriever
from rag_pipeline import RAGPipeline
from evaluate import evaluate_pipeline
from xapian_retriever import XapianRetriever
import numpy as np
from tqdm import tqdm

import torch
import wandb


train_file = "downloads/data/retriever/nq-train.json"
test_file = "downloads/data/retriever/nq-dev.json"

# ------------------ Load Data ------------------ #
def load_dpr_json(path):
    with open(path) as f:
        data = json.load(f)

    processed = []
    for item in data:
        query = item["question"]
        answers = item.get("answers", [])
        pos_ctxs = item.get("positive_ctxs", [])
        hard_neg_ctxs = item.get("hard_negative_ctxs", [])

        if not pos_ctxs:
            continue

        # Sort positives by score if available
        pos_ctxs_sorted = sorted(pos_ctxs, key=lambda x: x.get("score", 0), reverse=True)

        # Use top-1 positive
        top_pos = pos_ctxs_sorted[0]
        top_text = top_pos.get("text", "")

        # Use low-score positives as hard negatives (bottom 2, excluding top)
        weak_positives_as_negs = [
            ctx.get("text", "") for ctx in pos_ctxs_sorted[-2:]
            if ctx.get("text", "") != top_text
        ]

        # Use 2 hard negatives from hard_negative_ctxs
        hard_negs = [ctx.get("text", "") for ctx in hard_neg_ctxs[:2]]

        # Combine weak positives and hard negatives
        combined_negs = weak_positives_as_negs + hard_negs

        processed.append({
            "query": query,
            "positive_docs": top_text,
            "negative_docs": combined_negs,
            "answers": answers
        })

    return processed

train_data = load_dpr_json(train_file)
test_data = load_dpr_json(test_file)
print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")

batch_size = 8
stage1_epochs = 0
stage2_epochs = 2
stage3_epochs = 6

wandb.init(project="rag-qa", name="RAG_ThreeStage", config={
    "batch_size": batch_size,
    "stage1_epochs": stage1_epochs,
    "stage2_epochs": stage2_epochs,
    "stage3_epochs": stage3_epochs,
    "retriever_model": "intfloat/e5-large-v2",
    "generator_model": "facebook/bart-base",
    "optimizer": "AdamW",
    "retriever_lr": 2e-5,
    "generator_lr": 3e-5,
    "top_k": 50,
    "index": "wikipedia_faiss_index",
    "metadata": "wikipedia_metadata.jsonl",
},
notes="Three-stage fine-tuning: retriever, generator, end-to-end RAG. FAISS + E5 + T5.",                         
resume=True,                                 
)

wandb_table_train = wandb.Table(columns=["query", "answer", "retrieved_contexts"])
wandb_table_infer = wandb.Table(columns=["Query", "Answer", "TopDocs", "Prompt"])


def evaluate_retriever_recall(retriever, data, k=50):
    correct = 0
    total = 0
    for sample in data:
        query = sample["query"]
        positive = sample["positive"]
        retrieved = retriever.search(query, k)
        texts = [r[0].get("text", "") for r in retrieved]
        if positive in texts:
            correct += 1
        total += 1
    recall = correct / total if total > 0 else 0
    print(f"🔎 Retriever Recall@{k}: {recall:.3f}")
    return recall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Initialize retriever before stage 1
xapian = None

# Initialize retriever before stage 1
retriever = DenseRetriever(index_path="wikipedia_faiss_index",
                           metadata_path="wikipedia_metadata.jsonl",
                           device=device,
                           fine_tune=False)
# W&B watch: track retriever parameters and gradients
# wandb.watch(retriever.model, log="all", log_freq=100)


# print("🔹 Stage 1: Retriever Fine-tuning")
# retriever.enable_fine_tuning()
# for epoch in range(stage1_epochs):  # Adjust epochs if needed
#     print(f"\nRetriever Epoch {epoch+1}/{stage1_epochs}")
#     np.random.shuffle(train_data)
#     num_batches = len(train_data) // batch_size
#     batch_indices = range(0, len(train_data), batch_size)
#     with tqdm(total=num_batches, desc="Training Batches") as pbar:
#         for i in batch_indices:
#             batch = train_data[i: i + batch_size]
#             if len(batch) < batch_size:
#                 continue
#             loss = retriever.fine_tune_on_batch(batch)
#             pbar.update(1)
#             wandb.log({"retriever_loss": loss, "stage": "Stage1"})
# retriever.save("checkpoints/retriever")

stage2_epochs = 2
batch_size = 16
train_losses = []
eval_scores = []

total_steps = (len(train_data) // batch_size) * stage2_epochs * 2

# Stage 2: Generator-only fine-tuning
retriever.disable_fine_tuning()
rag = RAGPipeline(
    dense_retriever=retriever,
    device=device,
    fine_tune=True,
    alpha=0.2,
    total_steps=total_steps,
    inference_table_train=wandb_table_train,
    inference_table_infer=wandb_table_infer
    )
# W&B watch: track generator parameters and gradients
wandb.watch(rag.generator, log="all", log_freq=100)
wandb.watch(rag.dense_retriever.model, log="all", log_freq=100)

print("RAG pipeline initialized with DenseRetriever.")

for epoch in range(stage2_epochs):
    print(f"\n Epoch {epoch+1}/{stage2_epochs}")
    print("Training generator only...")
    rag.generator.train()
    rag.dense_retriever.model.eval()
    epoch_loss = rag.train(train_data, batch_size=batch_size, epochs=1, stage="Stage2")
    train_losses.extend(epoch_loss)


# Stage 3: End-to-End fine-tuning
print("\n🔄 Stage 3: End-to-End fine-tuning")
retriever.enable_fine_tuning()
rag._init_optimizer()  # Initialize optimizer for end-to-end training
# wandb.watch(rag.generator, log="all", log_freq=100)
# wandb.watch(rag.dense_retriever.model, log="all", log_freq=100)
for epoch in range(stage3_epochs):  # Reduce epochs for quicker fine-tuning
    print(f"\nEpoch {epoch+1} (End-to-End)")
    rag.generator.train()
    rag.dense_retriever.model.train()
    print("Training end-to-end...")
    epoch_loss = rag.train(train_data, batch_size=batch_size, epochs=1, stage="Stage3")

    train_losses.extend(epoch_loss)

    rag.generator.eval()
    rag.dense_retriever.model.eval()
    scores = evaluate_pipeline(rag, test_data, verbose=True, log_path=f"stage3_predictions_epoch{epoch+1}.json", top_k=50)
    eval_scores.append(scores)
    print(f"Eval — EM: {scores['EM']}%, F1: {scores['F1']}%")
    wandb.log({
        "EM": scores["EM"],
        "F1": scores["F1"],
        "stage": "Stage3",
        "epoch": epoch + 1
    })
    best_prev_f1 = max((s.get("F1", 0) for s in eval_scores[:-1]), default=0)
    if scores["F1"] > best_prev_f1:
        rag.generator.save_pretrained("best_model/generator")
        rag.tokenizer.save_pretrained("best_model/tokenizer")
        rag.dense_retriever.model.save("best_model/retriever")
        print("✅ Best model updated and saved to 'best_model'")


# Plotting
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.savefig("training_loss.png")

plt.figure()
plt.plot([s["F1"] for s in eval_scores], label='F1 Score', color='green')
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.title("F1 Score Over Epochs")
plt.legend()
plt.savefig("f1_score.png")

plt.figure()
plt.plot([s["EM"] for s in eval_scores], label='EM Score', color='green')
plt.xlabel("Epoch")
plt.ylabel("EM")
plt.title("EM Score Over Epochs")
plt.legend()
plt.savefig("EM_score.png")

# Save final
checkpoint_path = f"checkpoints/final_model"
os.makedirs(checkpoint_path, exist_ok=True)
rag.generator.save_pretrained("checkpoints/final_model/generator")
rag.tokenizer.save_pretrained("checkpoints/final_model/tokenizer")
rag.dense_retriever.model.save("checkpoints/final_model/retriever")
wandb.finish()
print(f"💾 Final model saved to '{checkpoint_path}'")
