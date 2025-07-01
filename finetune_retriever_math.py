import torch
import torch.optim as optim
import wandb
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, get_scheduler
from torch.utils.data import DataLoader
import pandas as pd

# --- Step 1: Import all custom components ---
# We will use the loader for the MATH dataset
from MathDataset import MathDataset, load_math_from_dir, collate_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "7" 


# --- Step 2: Define Evaluation Functions for the MATH Dataset ---
def extract_final_numeric_answer(text):
    text = str(text).replace(',', '')
    tokens = re.findall(r"-?\d+\.?\d*e[+\-]?\d+|-?\d+\.\d+|-?\d+", text)
    return tokens[-1] if tokens else ""

def extract_math_gold_answer(text: str) -> str:
    """
    From MATH 'solution' field (which may contain '\\boxed{<num>}'),
    extracts the gold numeric answer.
    """
    if isinstance(text, str):
        m = re.search(r"\\boxed\{(.+?)\}", text)
        if m:
            # The content inside boxed can be complex, so we parse it again
            return extract_final_numeric_answer(m.group(1))
        # Fallback to last numeric token if no \boxed{}
        return extract_final_numeric_answer(text)
    return str(text)

def compute_accuracy(preds, golds):
    correct = sum(1 for p, g in zip(preds, golds) if p.strip() == g.strip())
    return (correct / len(preds)) * 100 if preds else 0.0

def evaluate_generator(generator_model, eval_dataloader, generator_tokenizer, device):
    """
    Evaluates the generator by feeding it questions directly from the MATH dataset.
    """
    generator_model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Running SFT Evaluation on MATH"):
            input_ids = torch.stack(batch['input_ids']).to(device)
            attention_mask = torch.stack(batch['attention_mask']).to(device)
            gold_answers = batch['original_answer']
            
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
                all_golds.append(extract_math_gold_answer(gold_answers[i]))

    accuracy = compute_accuracy(all_preds, all_golds)
    return {"EM_accuracy": accuracy}

# --- Step 3: Main SFT Training Function ---
def main():
    config = {
        "generator_name": "facebook/bart-large-cnn",
        "train_dataset_path": "./thesis_datasets/math_hendrycks",
        "eval_dataset_path": "./thesis_datasets/math_hendrycks",
        "dataset_subjects": ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"],
        "num_epochs": 3, 
        "batch_size": 4,
        "learning_rate": 3e-5,
        "max_input_len": 512,
        "max_target_len": 512,
        "gradient_accumulation_steps": 4,
        "output_dir": "./finetuned_generator_direct_sft_math"
    }

    wandb.init(project="sft-generator-thesis", name="math-direct-finetuning", config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Model and Tokenizer ---
    print("--- Initializing Generator Model and Tokenizer ---")
    generator_tokenizer = AutoTokenizer.from_pretrained(config["generator_name"])
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(config["generator_name"]).to(device)

    # --- Load MATH Data ---
    train_data_list = load_math_from_dir(config["train_dataset_path"], config["dataset_subjects"])
    train_dataset = MathDataset(
        train_data_list, tokenizer=generator_tokenizer, max_q_len=config["max_input_len"], 
        max_a_len=config["max_target_len"], question_key='problem', answer_key='solution'
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=True)
    
    eval_data_list = load_math_from_dir(config["eval_dataset_path"], config["dataset_subjects"])
    eval_dataset = MathDataset(
        eval_data_list, tokenizer=generator_tokenizer, max_q_len=config["max_input_len"], 
        max_a_len=config["max_target_len"], question_key='problem', answer_key='solution'
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], collate_fn=collate_fn)

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(generator_model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, 
                                 num_training_steps=config["num_epochs"] * len(train_loader))
    
    # --- Training Loop ---
    print("\n--- Starting Direct Supervised Fine-Tuning on MATH ---")
    os.makedirs(config["output_dir"], exist_ok=True)
    best_accuracy = -1.0
    for epoch in range(config["num_epochs"]):
        generator_model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} Training")
        for step, batch in enumerate(progress_bar):
            
            input_ids = torch.stack(batch['input_ids']).to(device)
            attention_mask = torch.stack(batch['attention_mask']).to(device)
            labels = torch.stack(batch['labels']).to(device)

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
