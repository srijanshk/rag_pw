from unsloth import FastLanguageModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import wandb
import argparse
import re

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Llama 3.1 model for reasoning.")
    parser.add_argument("--model_name", type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct", help="Base model from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, default="./thesis_datasets/openmathinstruct2/openmathinstruct2_train_streamed.jsonl", help="Path to your local JSONL dataset file.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (r).")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Peak learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size.")
    parser.add_argument("--grad_accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions to generate per prompt for GRPO.")
    parser.add_argument("--wandb_project", type=str, default="active-retrieval", help="W&B project name.")
    parser.add_argument("--run_name", type=str, default="GRPO-Llama3.1-8B-run-1", help="A specific name for this W&B run.")
    return parser.parse_args()

SYSTEM_PROMPT = """You are a master mathematician. Solve the following problem by thinking step-by-step. If you need to retrieve information or use a tool, write `[SEARCH]`. Format your final response within <reasoning> and <answer> tags.

<reasoning>
Step-by-step thinking process.
</reasoning>
<answer>
The final numerical answer.
</answer>
"""

def extract_final_answer(text: str) -> str | None:
    """
    Extracts the final answer from a string, checking for both \\boxed{}
    and #### formats.
    """
    # Priority 1: Look for LaTeX \boxed{...} format
    # This pattern captures any character inside the braces.
    boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Priority 2: Look for #### format (if no \boxed{} is found)
    # This pattern captures the rest of the line after the delimiter.
    hash_match = re.search(r"####\s*(.*)", text)
    if hash_match:
        # Return the stripped content after the ####
        return hash_match.group(1).strip()

    # If neither format is found, return None
    return None

def format_dataset(ds: Dataset) -> Dataset:
    """
    Formats the OpenMathInstruct2 dataset for the GRPOTrainer.
    The trainer needs a 'prompt' column (list of dicts) and an 'answer' column.
    """
    def create_prompt_and_answer(row):
        question = row.get("problem")
        solution = row.get("generated_solution")
        ground_truth_answer = row.get("expected_answer") or extract_final_answer(solution)

        # Skip rows where we can't find a clear question or answer
        if not question or not ground_truth_answer:
            return None

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "answer": ground_truth_answer,
        }

    # Use .map() and then .filter() to handle failed extractions
    processed_ds = ds.map(create_prompt_and_answer, remove_columns=ds.column_names)
    filtered_ds = processed_ds.filter(lambda row: row['prompt'] is not None)
    return filtered_ds

# -----------------------------------
# --- 3. Reward Functions ---
# -----------------------------------
def extract_xml_answer(text: str) -> str:
    """Extracts content from the <answer> tag."""
    if "<answer>" not in text: return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Main reward: gives a high reward if the extracted answer is correct."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    ground_truth_answers = answer # The 'answer' column from our dataset
    rewards = []
    for resp, truth in zip(extracted_responses, ground_truth_answers):
        rewards.append(2.0 if resp == truth else 0.0)
    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """Bonus reward for adhering to the strict XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    # Use re.DOTALL to make '.' match newlines
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def main():
    args = get_args()
    print("🚀 Starting fine-tuning process with the following parameters:")

    # --- Initialize W&B ---
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # --- Load Model and Tokenizer via Unsloth ---
    print("🚀 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        max_lora_rank=args.lora_rank,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # --- Load and Prepare Dataset ---
    print(f"📚 Loading and preparing dataset from {args.dataset_path}...")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {args.dataset_path}")

    full_dataset = load_dataset("json", data_files={"train": args.dataset_path}, split="train")
    train_dataset = format_dataset(full_dataset)
    print(f"✅ Dataset prepared. Original size: {len(full_dataset)}, Filtered size: {len(train_dataset)}")


    # --- Configure GRPO Trainer ---
    print("⚙️ Configuring GRPOTrainer...")
    max_prompt_length = 512 # Max length for the prompt part
    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        optim="paged_adamw_8bit",
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        report_to="wandb",
        save_strategy="epoch",
        # GRPO-specific arguments
        max_steps = 500,
        save_steps = 500,
        max_grad_norm = 0.1,
        logging_steps = 1,
        num_generations=args.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=args.max_seq_length - max_prompt_length,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[
            correctness_reward_func,
            format_reward_func,
        ],
    )

    # --- Start Training ---
    print("💪 Starting training...")
    trainer.train()
    print("✅ Training complete!")

    # --- Save Final LoRA Adapters ---
    final_lora_path = f"./{args.wandb_project}_{wandb.run.id}_final_lora"
    model.save_pretrained(final_lora_path)
    tokenizer.save_pretrained(final_lora_path)
    print(f"📦 Final LoRA adapters saved to {final_lora_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
