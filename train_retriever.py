import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# 🧠 Config
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIR = "retriever_finetuned_st"
BATCH_SIZE = 32
EPOCHS = 5
SAVE_EVERY_STEPS = 10000  # Optional: useful for resuming

print("🔹 Loading Natural Questions...")
dataset = load_dataset("nq_open", split="train[:50%]")

# Step 2: Build InputExample list
print("🔹 Formatting examples...")
examples = []
for row in dataset:
    if row["answer"]:
        question = row["question"]
        answer = row["answer"][0]
        examples.append(InputExample(texts=[question, answer]))

print(f"✅ Total pairs: {len(examples)}")

# Step 3: Build SentenceTransformer with pooling
print("🔹 Building SentenceTransformer model...")
word_embedding_model = models.Transformer(MODEL_NAME)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Step 4: Define DataLoader + Loss
print("🔹 Setting up DataLoader and Loss...")
train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Step 5: Train
print("🚀 Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=1000,
    show_progress_bar=True,
    output_path=OUTPUT_DIR
)

print(f"✅ Done! Model saved to '{OUTPUT_DIR}'")
