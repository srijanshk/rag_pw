import string
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from pytorchtools import EarlyStopping
from sentence_transformers.util import batch_to_device

# Generator imports
from transformers import BartTokenizer, BartForConditionalGeneration

# Configuration
DATA_PATH = "downloads/data/retriever/nq-train.json"
VAL_PATH = "downloads/data/retriever/nq-dev.json"
MODEL_NAME = "intfloat/e5-large-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 10
SAVE_PATH = "models"

wandb.init(project="rag-retriever", name="fine-tune-e5", id="ougd4jsd",     
    resume="must", )

# Normalization and scoring functions for multi-answer EM/F1 evaluation
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return max([pred == normalize_answer(ans) for ans in gold_answers])

def f1_score(prediction, gold_answers):
    def score(gold, pred):
        gold_tokens = normalize_answer(gold).split()
        pred_tokens = normalize_answer(pred).split()
        common = set(gold_tokens) & set(pred_tokens)
        num_same = sum(min(gold_tokens.count(w), pred_tokens.count(w)) for w in common)
        if len(gold_tokens) == 0 or len(pred_tokens) == 0:
            return int(gold_tokens == pred_tokens)
        if num_same == 0:
            return 0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return (2 * precision * recall) / (precision + recall)
    return max([score(gold, prediction) for gold in gold_answers])

def load_dpr_data(path, max_negs=2):
    with open(path) as f:
        data = json.load(f)
    examples = []
    for row in data:
        query = f"query: {row['question']}"
        answers = row.get("answers", [""])
        for pos in row.get("positive_ctxs", [])[:1]:
            positive = f"passage: {pos['text']}"
            hard_negs = row.get("hard_negative_ctxs", [])
            negatives = []
            seen_ids = {pos.get("id")}
            for neg in hard_negs:
                if neg.get("id") not in seen_ids:
                    negatives.append(f"passage: {neg['text']}")
                    seen_ids.add(neg.get("id"))
                if len(negatives) >= max_negs:
                    break
            # Add low-score positive(s) as hard negatives if available
            low_score_positives = sorted(
                row.get("positive_ctxs", [])[1:], 
                key=lambda x: x.get("score", 0)
            )
            for lsp in low_score_positives:
                if lsp.get("id") not in seen_ids:
                    negatives.append(f"passage: {lsp['text']}")
                    seen_ids.add(lsp.get("id"))
                if len(negatives) >= max_negs + 2:
                    break
            examples.append({"query": query, "positive": positive, "negatives": negatives, "answer": answers[0]})
    return examples

def collate_fn(batch):
    queries = [b["query"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = sum([b["negatives"] for b in batch], [])  # flatten
    answers = [b["answer"] for b in batch]
    return queries, positives, negatives, answers

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
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)
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
    wandb.log({f"val/Recall@{k}": recall, f"val/MRR@{k}": mrr})
    for sub_k in [5, 10, 20, 100]:
        recall_k = sum([1 for rr in reciprocal_ranks if rr >= 1/sub_k]) / len(reciprocal_ranks)
        wandb.log({f"val/Recall@{sub_k}": recall_k})
    print(f"🔍 Recall@{k}: {recall:.4f}, MRR@{k}: {mrr:.4f}")
    return recall, mrr

def evaluate_with_gold_context(model, generator, tokenizer_gen, val_data, k=20):
    generator.eval()
    em_scores = []
    f1_scores = []

    with torch.no_grad():
        for sample in random.sample(val_data, min(500, len(val_data))):
            question = sample['question']
            answers = sample.get("answers", [])
            gold_ctx = sample['positive_ctxs'][0]['text']
            prompt = f"{question} {gold_ctx}"

            inputs = tokenizer_gen(
                prompt, return_tensors="pt", truncation=True, padding=True, max_length=512
            ).to(DEVICE)

            # Use beam search decoding
            outputs = generator.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                forced_bos_token_id=0  # Required for BART
            )
            generated = tokenizer_gen.decode(outputs[0], skip_special_tokens=True)

            em = exact_match(generated, answers)
            f1 = f1_score(generated, answers)
            em_scores.append(em)
            f1_scores.append(f1)

    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    wandb.log({f"val/EM@{k}_gold_ctx": avg_em, f"val/F1@{k}_gold_ctx": avg_f1})
    print(f"🧪 [GENERATOR] EM@{k} (Gold Context): {avg_em:.4f}, F1: {avg_f1:.4f}")
    return avg_em, avg_f1

class RetrieverTrainer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=EPOCHS * (len(load_dpr_data(DATA_PATH)) // BATCH_SIZE)
        )
        self.scaler_retriever = torch.amp.GradScaler()
        self.scaler_generator = torch.amp.GradScaler()
        self.temperature = 0.05

        # Generator initialization
        self.tokenizer_gen = BartTokenizer.from_pretrained("facebook/bart-base")
        self.generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(DEVICE)
        self.generator.train()
        self.gen_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=5e-6)
        self.gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.gen_optimizer,
            T_max=EPOCHS * (len(load_dpr_data(DATA_PATH)) // BATCH_SIZE)
        )
        self.last_batch_examples = None

    def train_step(self, queries, positives, negatives):

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
            # Use in-batch negatives by default
            labels = torch.arange(q_emb.size(0), device=DEVICE)
            loss = F.cross_entropy(sim_matrix, labels)

        self.scaler_retriever.scale(loss).backward()
        # (Gradient unscale, clipping, and logging moved to main loop)
        return loss.item()

    def save(self, path):
        self.model.save(path)

    def generator_step(self, queries, answers, contexts):
        self.generator.train()
        prompts = [f"{q.replace('query: ', '')} {c.replace('passage: ', '')}" for q, c in zip(queries, contexts)]
        inputs = self.tokenizer_gen(
            prompts, padding=True, truncation=True, return_tensors='pt', max_length=512
        ).to(DEVICE)
        targets = self.tokenizer_gen(
            answers, padding=True, truncation=True, return_tensors='pt', max_length=64
        ).input_ids.to(DEVICE)

        labels = targets.clone()
        labels[labels == self.tokenizer_gen.pad_token_id] = -100

        with torch.amp.autocast(device_type='cuda'):
            output = self.generator(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            # Apply label smoothing to generator loss
            # logits = output.logits
            # loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss = output.loss

        self.scaler_generator.scale(loss).backward()
        return loss.item()

    def generator_step_with_retriever(self, queries, answers):
        # Retrieve top-3 contexts for each query and concatenate them for generation
        self.model.eval()
        self.generator.train()
        prompts = []
        with torch.no_grad():
            query_embs = self.model.encode(queries, convert_to_tensor=True, device=DEVICE, normalize_embeddings=True)
            all_passages = []
            for ex in self.last_batch_examples:
                all_passages.append(ex["positive"])
                all_passages.extend(ex["negatives"])
            passage_embs = self.model.encode(all_passages, convert_to_tensor=True, device=DEVICE, normalize_embeddings=True)
            sims = torch.matmul(query_embs, passage_embs.T)
            top_indices = torch.topk(sims, k=3, dim=1).indices.tolist()
            retrieved_passages = [" ".join([all_passages[i] for i in top_k]) for top_k in top_indices]
            prompts = [
                f"{q.replace('query:', '').strip()} {p.replace('passage:', '').strip()}"
                for q, p in zip(queries, retrieved_passages)
            ]

        inputs = self.tokenizer_gen(
            prompts, padding=True, truncation=True, return_tensors='pt', max_length=512
        ).to(DEVICE)
        targets = self.tokenizer_gen(
            answers, padding=True, truncation=True, return_tensors='pt', max_length=64
        ).input_ids.to(DEVICE)

        labels = targets.clone()
        labels[labels == self.tokenizer_gen.pad_token_id] = -100

        # Skip batch if all labels are masked
        if (labels != -100).sum().item() == 0:
            return 0.0  # skip batch if all labels are masked

        with torch.amp.autocast(device_type='cuda'):
            output = self.generator(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            loss = output.loss

        # Skip batch if loss is not finite
        if not torch.isfinite(loss):
            print(f"⚠️ NaN loss detected. Skipping this batch.")
            return 0.0

        self.scaler_generator.scale(loss).backward()
        # Gradient unscale, clipping, and logging moved to main loop
        return loss.item()

def evaluate_with_retrieved_context(model, generator, tokenizer_gen, val_data, k=20):
    generator.eval()
    em_scores = []
    f1_scores = []
    with torch.no_grad():
        for sample in random.sample(val_data, min(500, len(val_data))):
            query = f"query: {sample['question']}"
            all_ctxs = sample.get("positive_ctxs", []) + sample.get("hard_negative_ctxs", [])
            if not all_ctxs:
                continue
            passages = [ctx["text"] for ctx in all_ctxs]
            p_embs = model.encode(passages, convert_to_tensor=True, device=DEVICE, normalize_embeddings=True)
            q_emb = model.encode(query, convert_to_tensor=True, device=DEVICE, normalize_embeddings=True)
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            scores = torch.matmul(p_embs, q_emb.T).squeeze(-1)
            top_passage = passages[scores.argmax().item()]

            prompt = f"{sample['question']} {top_passage}"
            inputs = tokenizer_gen(prompt, return_tensors='pt', truncation=True, padding=True, max_length=512).to(DEVICE)
            # Use beam search decoding
            outputs = generator.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                forced_bos_token_id=0  # Required for BART
            )

            generated = tokenizer_gen.decode(outputs[0], skip_special_tokens=True)

            em = exact_match(generated, sample["answers"])
            f1 = f1_score(generated, sample["answers"])
            em_scores.append(em)
            f1_scores.append(f1)

    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    wandb.log({f"val/EM@{k}_retrieved_ctx": avg_em, f"val/F1@{k}_retrieved_ctx": avg_f1})
    print(f"📘 [GENERATOR] EM@{k} (Retrieved Context): {avg_em:.4f}, F1: {avg_f1:.4f}")
    return avg_em, avg_f1

def main():
    train_data = load_dpr_data(DATA_PATH)
    val_data = json.load(open(VAL_PATH))

    wandb.define_metric("Recall@20", summary="max")
    wandb.define_metric("MRR@20", summary="max")
    wandb.define_metric("EM@20_retrieved_ctx", summary="max")
    wandb.define_metric("F1@20_retrieved_ctx", summary="max")

    trainer = RetrieverTrainer(MODEL_NAME)
    best_mrr = 0.0

    # Initialize EarlyStopping before training loop
    early_stopper = EarlyStopping(patience=2, verbose=True, path=f"{SAVE_PATH}/retriever_finetuned_e5_earlystop.pt")

    for epoch in range(EPOCHS):
        random.shuffle(train_data)
        loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        epoch_loss = 0
        epoch_gen_loss = 0
        for step, (queries, positives, negatives, answers) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            loss = trainer.train_step(queries, positives, negatives)
            epoch_loss += loss
            wandb.log({
                "train/loss": loss,
                "train/epoch": epoch + 1,
                "train/step": step + 1
            })

            trainer.last_batch_examples = loader.dataset[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            gen_loss = trainer.generator_step_with_retriever(queries, answers)
            epoch_gen_loss += gen_loss
            wandb.log({
                "train/generator_loss": gen_loss,
                "train/epoch": epoch + 1,
                "train/step": step + 1
            })

            if (step + 1) % 4 == 0:
                # Unscale & clip gradients for retriever
                trainer.scaler_retriever.unscale_(trainer.optimizer)
                retriever_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
                wandb.log({"train/retriever_grad_norm": retriever_norm.item()})

                # Unscale & clip gradients for generator (before optimizer step)
                trainer.scaler_generator.unscale_(trainer.gen_optimizer)
                gen_norm = torch.nn.utils.clip_grad_norm_(trainer.generator.parameters(), max_norm=1.0)
                wandb.log({"train/generator_grad_norm": gen_norm.item()})

                trainer.scaler_retriever.step(trainer.optimizer)
                trainer.scaler_retriever.update()
                trainer.scaler_generator.step(trainer.gen_optimizer)
                trainer.scaler_generator.update()
                trainer.optimizer.zero_grad()
                trainer.gen_optimizer.zero_grad()
                trainer.scheduler.step()
                trainer.gen_scheduler.step()
                wandb.log({
                    "train/lr": trainer.scheduler.get_last_lr()[0],
                    "train/gen_lr": trainer.gen_scheduler.get_last_lr()[0]
                })
        avg_gen_loss = epoch_gen_loss / len(loader)
        wandb.log({"train/epoch_generator_loss": avg_gen_loss, "train/epoch": epoch + 1})
        print(f"✅ Epoch {epoch+1} Avg Generator Loss: {avg_gen_loss:.4f}")
        wandb.log({"train/epoch_loss": epoch_loss / len(loader), "train/epoch": epoch + 1})

        print(f"✅ Epoch {epoch+1} Avg Loss: {epoch_loss / len(loader):.4f}")
        _, mrr = evaluate(trainer.model, val_data)
        evaluate_with_gold_context(trainer.model, trainer.generator, trainer.tokenizer_gen, val_data)
        evaluate_with_retrieved_context(trainer.model, trainer.generator, trainer.tokenizer_gen, val_data)
        if mrr > best_mrr:
            best_mrr = mrr
            trainer.save(f"{SAVE_PATH}/retriever_finetuned_e5_best")
            trainer.generator.save_pretrained(f"{SAVE_PATH}/generator_best")
            trainer.tokenizer_gen.save_pretrained(f"{SAVE_PATH}/generato2r_best")

        # Early stopping step
        early_stopper(mrr, trainer.model)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            break

if __name__ == "__main__":
    main()
