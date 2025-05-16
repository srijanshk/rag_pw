import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import argparse
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
# from sklearn.metrics import f1_score
import os
import wandb
from collections import Counter

def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 between prediction and ground truth.
    """
    pred_tokens = prediction.split()
    gold_tokens = ground_truth.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    samples = []
    for item in j.get('data', []):
        if not item.get('short_answers'):
            continue
        samples.append({
            'query':            item['question'],
            'positive_context': item['context'],
            'answer':           item['short_answers'][0]
        })
    return samples

class GoldContextDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        inp = self.tokenizer(
            f"question: {s['query']}  context: {s['positive_context']}",
            max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        )
        tgt = self.tokenizer(
            s['answer'], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        )
        labels = tgt.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids':      inp.input_ids.squeeze(0),
            'attention_mask': inp.attention_mask.squeeze(0),
            'labels':         labels.squeeze(0)
        }

def evaluate(model, tokenizer, dataset, device):
    model.eval()
    preds, golds = [], []
    loader = DataLoader(dataset, batch_size=8)
    with torch.no_grad():
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items() if k!='labels'}
            out = model.generate(**batch, max_length=64)
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            preds.extend(texts)
        # extract golds
        for s in dataset.samples:
            golds.append(s['answer'])
    model.train()
    f1s = []
    for p, g in zip(preds, golds):
        f1s.append(compute_f1(p, g))
    return sum(f1s) / len(f1s) if f1s else 0.0

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    model     = BartForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
    wandb.watch(model, log="all", log_freq=100)

    train_samples = load_data(args.train_path)
    dev_samples   = load_data(args.dev_path)
    train_ds = GoldContextDataset(train_samples, tokenizer, args.max_length)
    dev_ds   = GoldContextDataset(dev_samples,   tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    best_dev_f1 = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):
            optimizer.zero_grad()
            batch = {k:v.to(device) for k,v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            # log each batch loss to W&B
            wandb.log({
                'batch_loss': loss.item(),
                'epoch': epoch,
                'batch_idx': batch_idx
            })
            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        dev_f1   = evaluate(model, tokenizer, dev_ds, device)
        # log metrics to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'dev_f1': dev_f1
        })
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            wandb.log({'best_dev_f1': best_dev_f1})
            print(f"Saved new best model (F1={best_dev_f1:.4f}) to {args.output_dir}")
        print(f"Epoch {epoch}  Train loss: {avg_loss:.4f}  Dev F1: {dev_f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path',   type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs',     type=int, default=3)
    parser.add_argument('--lr',         type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='best_model_generator',
                        help='Directory to save the best model')
    parser.add_argument('--wandb_project', type=str, default='rag_generator',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity',  type=str, default=None,
                        help='Weights & Biases entity (team or user)')
    args = parser.parse_args()
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )
    train(args)