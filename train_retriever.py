import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from tqdm import tqdm
import wandb
from torch.cuda.amp import GradScaler

# Configuration
DATA_PATH = "downloads/data/retriever/nq-train.json"
VAL_PATH = "downloads/data/retriever/nq-dev.json"
MODEL_NAME = "intfloat/e5-large-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 3
SAVE_PATH = "retriever_finetuned_e5"

wandb.init(project="rag-retriever", name="fine-tune-e5")

def load_dpr_data(path, max_negs=2):
    with open(path) as f:
        data = json.load(f)
    examples = []
    for row in data:
        query = f"query: {row['question']}"
        for pos in row.get("positive_ctxs", [])[:1]:
            positive = f"passage: {pos['text']}"
            negatives = [
                f"passage: {neg['text']}" 
                for neg in row.get("negative_ctxs", [])[:max_negs]
                if neg.get("id") != pos.get("id")
            ]
            examples.append({"query": query, "positive": positive, "negatives": negatives})
    return examples

def collate_fn(batch):
    queries = [b["query"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = sum([b["negatives"] for b in batch], [])  # flatten
    return queries, positives, negatives

def evaluate(model, val_data, k=20):
    model.eval()
    correct = 0
    total = 0
    reciprocal_ranks = []

    with torch.no_grad():
        for sample in random.sample(val_data, min(500, len(val_data))):
            query = f"query: {sample['question']}"
            true_id = str(sample["positive_ctxs"][0].get("passage_id", sample["positive_ctxs"][0].get("id")))

            query_emb = model.encode(query, convert_to_tensor=True, device=DEVICE)
            passages = [f"passage: {ctx['text']}" for ctx in sample["negative_ctxs"][:20]]
            passages.append(f"passage: {sample['positive_ctxs'][0]['text']}")

            passage_ids = [str(ctx.get("passage_id", ctx.get("id"))) for ctx in sample["negative_ctxs"][:20]]
            passage_ids.append(true_id)

            p_embs = model.encode(passages, convert_to_tensor=True, device=DEVICE)
            query_emb = F.normalize(query_emb, p=2, dim=1)
            p_embs = F.normalize(p_embs, p=2, dim=1)
            sims = torch.matmul(p_embs, query_emb.T).squeeze(-1)

            sorted_ids = [x for _, x in sorted(zip(sims.tolist(), passage_ids), reverse=True)]

            if true_id in sorted_ids:
                rank = sorted_ids.index(true_id) + 1
                reciprocal_ranks.append(1.0 / rank)
                correct += 1
            else:
                reciprocal_ranks.append(0.0)
            total += 1

    recall = correct / total
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    wandb.log({f"Recall@{k}": recall, f"MRR@{k}": mrr})
    for sub_k in [5, 10, 20, 100]:
        recall_k = sum([1 for rr in reciprocal_ranks if rr >= 1/sub_k]) / len(reciprocal_ranks)
        wandb.log({f"Recall@{sub_k}": recall_k})
    print(f"🔍 Recall@{k}: {recall:.4f}, MRR@{k}: {mrr:.4f}")
    return recall, mrr

class RetrieverTrainer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=EPOCHS * (len(load_dpr_data(DATA_PATH)) // BATCH_SIZE)
        )
        self.scaler = torch.amp.GradScaler()
        self.temperature = 0.05

    def train_step(self, queries, positives, negatives):
        from sentence_transformers.util import batch_to_device

        all_queries = queries
        all_passages = positives + negatives
        self.model.train()
        tokenizer = self.model.tokenizer
        queries_batch = tokenizer(all_queries, padding=True, truncation=True, return_tensors='pt')
        passages_batch = tokenizer(all_passages, padding=True, truncation=True, return_tensors='pt')

        queries_batch = batch_to_device(queries_batch, DEVICE)
        passages_batch = batch_to_device(passages_batch, DEVICE)

        with torch.amp.autocast(device_type='cuda'):
            q_output = self.model.forward(queries_batch)
            p_output = self.model.forward(passages_batch)
            q_emb = F.normalize(q_output['sentence_embedding'], p=2, dim=1)
            p_emb = F.normalize(p_output['sentence_embedding'], p=2, dim=1)

            sim_matrix = torch.matmul(q_emb, p_emb.T) / self.temperature
            labels = torch.arange(len(all_queries), device=DEVICE)
            loss = F.cross_entropy(sim_matrix, labels)

        self.scaler.scale(loss).backward()
        return loss.item()

    def save(self, path):
        self.model.save(path)

def main():
    train_data = load_dpr_data(DATA_PATH)
    val_data = json.load(open(VAL_PATH))

    wandb.define_metric("Recall@20", summary="max")
    wandb.define_metric("MRR@20", summary="max")

    trainer = RetrieverTrainer(MODEL_NAME)
    best_mrr = 0.0

    for epoch in range(EPOCHS):
        random.shuffle(train_data)
        loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        epoch_loss = 0
        for step, (queries, positives, negatives) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            loss = trainer.train_step(queries, positives, negatives)
            epoch_loss += loss
            wandb.log({"loss": loss})
            if (step + 1) % 4 == 0:
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
                trainer.optimizer.zero_grad()
                trainer.scheduler.step()
        print(f"✅ Epoch {epoch+1} Avg Loss: {epoch_loss / len(loader):.4f}")
        _, mrr = evaluate(trainer.model, val_data)
        if mrr > best_mrr:
            best_mrr = mrr
            trainer.save(f"{SAVE_PATH}_best")
        trainer.save(f"{SAVE_PATH}_epoch{epoch+1}")

if __name__ == "__main__":
    main()
