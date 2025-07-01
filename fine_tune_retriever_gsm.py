import torch
import torch.optim as optim
import wandb
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, get_scheduler
from torch.utils.data import DataLoader
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "6" 


# --- Step 1: Import all custom components ---
# We only need the MathDataset and its loader for this direct SFT task.
from MathDataset import MathDataset, load_gsm8k_from_file, collate_fn

# --- Step 2: Define Evaluation Functions ---
def extract_final_numeric_answer(text):
    text = str(text).replace(',', '')
    tokens = re.findall(r"-?\d+\.\d+|-?\d+", text)
    return tokens[-1] if tokens else ""

def extract_gsm8k_gold_answer(text):
    if isinstance(text, str):
        m = re.search(r"####\s*([-\d\.,]+)", text)
        if m: return m.group(1).replace(',', '')
        tokens = re.findall(r"-?\d+\.\d+|-?\d+", text.replace(',', ''))
        return tokens[-1] if tokens else ""
    return str(text)

def compute_accuracy(preds, golds):
    correct = sum(1 for p, g in zip(preds, golds) if p.strip() == g.strip())
    return (correct / len(preds)) * 100 if preds else 0.0

def evaluate_generator(generator_model, eval_dataloader, generator_tokenizer, device):
    """
    Evaluates the generator by feeding it questions directly.
    """
    generator_model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Running SFT Evaluation"):
            # Input to the model is now just the question, not context.
            # The 'input_ids' from the dataloader are the tokenized questions.
            input_ids = torch.stack(batch['input_ids']).to(device)
            attention_mask = torch.stack(batch['attention_mask']).to(device)
            gold_answers = batch['original_answer']
            
            # Generate answers directly from the questions
            generated_ids = generator_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512, 
                num_beams=4, 
                early_stopping=True
            )
            gen_answers = generator_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i in range(len(gen_answers)):
                all_preds.append(extract_final_numeric_answer(gen_answers[i]))
                all_golds.append(extract_gsm8k_gold_answer(gold_answers[i]))

    accuracy = compute_accuracy(all_preds, all_golds)
    return {"EM_accuracy": accuracy}

# --- Step 3: Main SFT Training Function ---
def main():
    config = {
        "generator_name": "facebook/bart-large-cnn",
        "train_dataset_path": "./thesis_datasets/gsm8k/train.jsonl",
        "eval_dataset_path": "./thesis_datasets/gsm8k/test.jsonl",
        "num_epochs": 3, 
        "batch_size": 4, # You can try increasing this as we are not using the retriever
        "learning_rate": 3e-5,
        "max_input_len": 512,  # Max length for the input question
        "max_target_len": 512, # Max length for the target answer
        "gradient_accumulation_steps": 4,
        "output_dir": "./finetuned_generator_direct_sft" # New output directory
    }

    wandb.init(project="sft-generator-thesis", name="gsm8k-direct-finetuning", config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Model and Tokenizer (Retriever is removed) ---
    print("--- Initializing Generator Model and Tokenizer ---")
    generator_tokenizer = AutoTokenizer.from_pretrained(config["generator_name"])
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(config["generator_name"]).to(device)

    # --- Load Data ---
    # The MathDataset will now just handle tokenizing the question and answer directly
    train_data_list = load_gsm8k_from_file(config["train_dataset_path"])
    train_dataset = MathDataset(
        train_data_list, tokenizer=generator_tokenizer, max_q_len=config["max_input_len"], 
        max_a_len=config["max_target_len"], question_key='question', answer_key='answer'
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=True)
    
    eval_data_list = load_gsm8k_from_file(config["eval_dataset_path"])
    eval_dataset = MathDataset(
        eval_data_list, tokenizer=generator_tokenizer, max_q_len=config["max_input_len"], 
        max_a_len=config["max_target_len"], question_key='question', answer_key='answer'
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], collate_fn=collate_fn)

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(generator_model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, 
                                 num_training_steps=config["num_epochs"] * len(train_loader))
    
    # --- Training Loop ---
    print("\n--- Starting Direct Supervised Fine-Tuning ---")
    os.makedirs(config["output_dir"], exist_ok=True)
    best_accuracy = -1.0
    for epoch in range(config["num_epochs"]):
        generator_model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} Training")
        for step, batch in enumerate(progress_bar):
            
            # --- The input is now directly the tokenized question from the dataloader ---
            input_ids = torch.stack(batch['input_ids']).to(device)
            attention_mask = torch.stack(batch['attention_mask']).to(device)
            labels = torch.stack(batch['labels']).to(device)

            # SFT Forward Pass
            outputs = generator_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config["gradient_accumulation_steps"]
            progress_bar.set_postfix({'loss': loss.item() * config["gradient_accumulation_steps"]})
            wandb.log({"sft_step_loss": loss.item() * config["gradient_accumulation_steps"]})

        # --- End of Epoch ---
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        # --- Evaluation ---
        eval_metrics = evaluate_generator(generator_model, eval_loader, generator_tokenizer, device)
        accuracy = eval_metrics['EM_accuracy']
        print(f"Epoch {epoch+1} Evaluation Accuracy: {accuracy:.2f}%")
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss, "eval_accuracy": accuracy})
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print("New best accuracy! Saving model...")
            save_dir = os.path.join(config["output_dir"], "best_model")
            os.makedirs(save_dir, exist_ok=True)
            generator_model.save_pretrained(save_dir)
            generator_tokenizer.save_pretrained(save_dir)

    wandb.finish()
    print("--- SFT Training Finished ---")

if __name__ == "__main__":
    main()
