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


import torch


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

        positives = item.get("positive_ctxs", [])
        negatives = item.get("negative_ctxs", [])

        # Extract text from positives and negatives
        pos_passage = positives[0]["text"] if positives else ""
        neg_passages = [neg["text"] for neg in negatives]

        processed.append({
            "query": query,
            "positive": pos_passage,
            "negatives": neg_passages,
            "answers": answers
        })
    return processed

train_data = load_dpr_json(train_file)
test_data = load_dpr_json(test_file)
print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")
# ------------------ Load Retriever ------------------ #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


print("🚀 Stage 1: Fine-tuning Generator Only")
stage1_epochs = 2
stage2_epochs = 4
batch_size = 128
train_losses = []
eval_scores = []

total_steps = (len(train_data) // batch_size) * (stage1_epochs * 2 + stage2_epochs * 2)

# Stage 1: Fine-tune generator only (retriever frozen)
retriever = DenseRetriever(index_path="wikipedia_faiss_index",
                           metadata_path="wikipedia_metadata.jsonl",
                           device=device,
                           fine_tune=False)  # Freezed
xapian = None
rag = RAGPipeline(dense_retriever=retriever, device=device, fine_tune=True, alpha=0.2, total_steps=total_steps)


for epoch in range(stage1_epochs):
    print(f"\n[Stage 1] Epoch {epoch+1}/{stage1_epochs}")
    epoch_loss = rag.train(train_data, batch_size=batch_size, epochs=1)
    train_losses.extend(epoch_loss)

    scores = evaluate_pipeline(rag, test_data, verbose=True, log_path=f"stage1_predictions_epoch{epoch+1}.json")
    eval_scores.append(scores)
    print(f"Eval — EM: {scores['EM']}%, F1: {scores['F1']}%")

# Stage 2: End-to-end fine-tuning (generator + retriever)
rag.fine_tune = True

rag.generator.train()

print("\n🚀 Stage 2: End-to-End Fine-tuning (Generator + Retriever)")

for epoch in range(stage2_epochs):
    print(f"\n[Stage 2] Epoch {epoch+1}/{stage2_epochs}")

    if epoch == 0:
        retriever.fine_tune = False  # Keep retriever frozen
    else:
        retriever.enable_fine_tuning()

    rag.fine_tune = True
    rag.generator.train()

    epoch_loss = rag.train(train_data, batch_size=batch_size, epochs=1)
    train_losses.extend(epoch_loss)
    scores = evaluate_pipeline(rag, test_data, verbose=True, log_path=f"stage2_predictions_epoch{epoch+1}.json")
    eval_scores.append(scores)
    print(f"Eval — EM: {scores['EM']}%, F1: {scores['F1']}%")

    if epoch == 0 or scores["F1"] > max(s.get("F1", 0) for s in eval_scores[:-1]):
        rag.generator.save_pretrained("best_model")
        rag.tokenizer.save_pretrained("best_model")
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
rag.generator.save_pretrained(checkpoint_path)
rag.tokenizer.save_pretrained(checkpoint_path)
print(f"💾 Final model saved to '{checkpoint_path}'")
