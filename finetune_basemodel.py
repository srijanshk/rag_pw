import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_PROJECT"] = "llama3.1-8b-baseline-thinking"  
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging as hf_logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import wandb # Import W&B

# Suppress some Hugging Face warnings for cleaner output
hf_logging.set_verbosity_warning()

# --- 1. Configuration ---
# Dataset Paths
INPUT_JSONL_FILE = "./thesis_datasets/openmathinstruct2/openmathinstruct2_train_streamed.jsonl"

# Model IDs
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR_BASELINE = "./results_llama3_1_8b_baseline_thinking_v3_wandb" # Updated output directory
ADAPTER_SAVE_PATH_BASELINE = os.path.join(OUTPUT_DIR_BASELINE, "adapter_model_baseline")
NUM_TRAIN_EPOCHS = 1

# W&B Configuration
WANDB_RUN_NAME = f"baseline-llama3.1-8b-{BASE_MODEL_ID.split('/')[-1]}-openmath-ep{NUM_TRAIN_EPOCHS}" # Dynamic run name

# QLoRA parameters
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

# TrainingArguments parameters
MAX_STEPS = 300 # For quick testing
FP16_OR_BF16_TRAINING = True
OUTPUT_DIR_TRAINER_BASELINE = os.path.join(OUTPUT_DIR_BASELINE, "trainer_checkpoints_baseline")
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 0.3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
OPTIMIZER_TYPE = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = True
SAVE_STRATEGY = "steps"
SAVE_STEPS = 200
LOGGING_STEPS = 10

# SFTTrainer parameters
MAX_SEQ_LENGTH = 2048
PACKING = False

# --- 2. Check GPU and Set Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    if BNB_4BIT_COMPUTE_DTYPE == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("⚠️ Warning: bfloat16 compute dtype for BitsAndBytes not supported. Falling back to float32.")
        BNB_4BIT_COMPUTE_DTYPE = "float32"
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU. QLoRA (4-bit) will be disabled. Training will be extremely slow.")
    if USE_4BIT: USE_4BIT = False
    FP16_OR_BF16_TRAINING = False

# --- 3. Dataset Loading and Preparation ---
print(f"\n🔄 Loading dataset from {INPUT_JSONL_FILE}...")
full_dataset = load_dataset('json', data_files=INPUT_JSONL_FILE, split='train')

print("ℹ️ Creating train/validation split (95%/5%)...")
train_test_split = full_dataset.train_test_split(test_size=0.05, seed=42)
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

print(f"📊 Dataset loaded. Training examples: {len(dataset_dict['train'])}")
print(f"📊 Validation examples: {len(dataset_dict['validation'])}")
if len(dataset_dict['train']) == 0:
    print("❌ FATAL: No training data. Check INPUT_JSONL_FILE."); exit()

# --- 4. Tokenizer ---
print(f"\n🔄 Loading tokenizer for {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Data Formatting Function for BASELINE ---
def format_for_baseline_thinking_model(example):
    problem_text = example.get('problem', '')
    reasoning_steps = example.get('generated_solution', '')
    expected_answer_text = example.get('expected_answer', '')
    assistant_content = (
        "<thinking>\n<reasoning>\n"
        f"{reasoning_steps}\n</reasoning>\n<output>\n"
        f"{expected_answer_text}\n</output>\n</thinking>")
    messages = [{"role": "user", "content": problem_text}, {"role": "assistant", "content": assistant_content}]
    try:
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    except Exception as e:
        print(f"❌ Error applying chat template: {e} on problem: {problem_text[:50]}..."); return {"text": ""}

print("\n🔄 Formatting dataset for BASELINE model...")
formatted_datasets = DatasetDict()
for split, ds in dataset_dict.items():
    formatted_datasets[split] = ds.map(format_for_baseline_thinking_model, remove_columns=list(ds.features))
    formatted_datasets[split] = formatted_datasets[split].filter(lambda example: len(example.get('text', '')) > 0)
    print(f"📊 {split} split formatted. Examples remaining: {len(formatted_datasets[split])}")

if len(formatted_datasets['train']) == 0: print("❌ FATAL: No training data after formatting."); exit()
print(f"📜 First formatted training example:\n{formatted_datasets['train'][0]['text']}")

# --- 5. Model Loading (QLoRA) ---
print(f"\n🔄 Loading base model {BASE_MODEL_ID} with QLoRA...")
compute_dtype_torch = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype_torch, bnb_4bit_use_double_quant=USE_NESTED_QUANT)
    print("✅ 4-bit quantization (QLoRA) enabled.")
else: print("ℹ️ 4-bit quantization (QLoRA) is NOT enabled.")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
model.config.use_cache = False

# --- 6. PEFT Configuration (LoRA) ---
print("\n🛠️ Setting up LoRA configuration...")
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, r=LORA_R, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

# --- 7. Training Arguments with W&B Integration ---
print("\n⚙️ Defining training arguments with W&B integration...")

# Check if WANDB_PROJECT is set, otherwise HuggingFace Trainer might prompt or use a default.
if os.getenv('WANDB_PROJECT') is None:
    print("⚠️ WANDB_PROJECT environment variable not set. Consider setting it for better organization in W&B.")

# training_args_dict = {
#     "output_dir": OUTPUT_DIR_TRAINER_BASELINE,
#     "num_train_epochs": NUM_TRAIN_EPOCHS,
#     "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
#     "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
#     "optim": OPTIMIZER_TYPE,
#     "save_strategy": SAVE_STRATEGY,
#     "save_steps": SAVE_STEPS,
#     "logging_strategy": "steps", # Log metrics at each logging_steps
#     "logging_steps": LOGGING_STEPS,
#     "learning_rate": LEARNING_RATE,
#     "weight_decay": WEIGHT_DECAY,
#     "fp16": FP16_OR_BF16_TRAINING and (BNB_4BIT_COMPUTE_DTYPE != "bfloat16"),
#     "bf16": FP16_OR_BF16_TRAINING and (BNB_4BIT_COMPUTE_DTYPE == "bfloat16"),
#     "max_grad_norm": MAX_GRAD_NORM,
#     "warmup_ratio": WARMUP_RATIO,
#     "group_by_length": GROUP_BY_LENGTH,
#     "lr_scheduler_type": LR_SCHEDULER_TYPE,
#     "report_to": "wandb",  # Enable W&B reporting
#     "run_name": WANDB_RUN_NAME, # Set a name for the W&B run
# }
config_args = {
    # --- SFT-specific arguments ---
    "max_seq_length": MAX_SEQ_LENGTH,
    "dataset_text_field": "text",
    "packing": PACKING,

    # --- Regular Training Arguments ---
    "output_dir": OUTPUT_DIR_TRAINER_BASELINE,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "optim": OPTIMIZER_TYPE,
    "save_strategy": "steps",
    "save_steps": SAVE_STEPS,
    "logging_strategy": "steps",
    "logging_steps": LOGGING_STEPS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "fp16": FP16_OR_BF16_TRAINING and (BNB_4BIT_COMPUTE_DTYPE != "bfloat16"),
    "bf16": FP16_OR_BF16_TRAINING and (BNB_4BIT_COMPUTE_DTYPE == "bfloat16"),
    "max_grad_norm": MAX_GRAD_NORM,
    "warmup_ratio": WARMUP_RATIO,
    "group_by_length": GROUP_BY_LENGTH,
    "lr_scheduler_type": LR_SCHEDULER_TYPE,
    "report_to": "wandb",
    "run_name": WANDB_RUN_NAME,
    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
}
# Add evaluation strategy if validation set exists
if 'validation' in formatted_datasets and len(formatted_datasets['validation']) > 0:
    config_args["eval_strategy"] = "steps"
    config_args["eval_steps"] = SAVE_STEPS # Evaluate at the same frequency as saving checkpoints
    # training_args_dict["load_best_model_at_end"] = True # Optional: load the best model at the end of training based on eval loss

# Check if MAX_STEPS is defined (e.g. not commented out in the config section)
try:
    if MAX_STEPS > 0 : config_args["max_steps"] = MAX_STEPS
except NameError: # MAX_STEPS was not defined (e.g. commented out)
    pass

sft_config = SFTConfig(**config_args)

if GRADIENT_CHECKPOINTING:
    sft_config.gradient_checkpointing = True
    # Some models / PyTorch versions might need use_reentrant=False for grad checkpointing
    sft_config.gradient_checkpointing_kwargs = {"use_reentrant": False}


# --- 8. SFTTrainer ---
print("\n🏋️ Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=formatted_datasets['train'],
    eval_dataset=formatted_datasets.get('validation'),
    peft_config=peft_config,
    processing_class=tokenizer,  
)

# --- 9. Training ---
print("\n🚀 Starting baseline model training with W&B logging...")
try:
    trainer.train()
    print("✅ Baseline model training finished.")
except Exception as e:
    print(f"❌ Training failed: {e}")
    # It's good practice to finish the W&B run even if training fails
    if os.getenv('WANDB_PROJECT'): # Check if W&B was intended to be used
        wandb.finish(exit_code=1, quiet=True) # Mark run as failed
    raise # Re-raise the exception

# --- 10. Save Model (Adapter and Tokenizer) ---
print(f"\n💾 Saving fine-tuned baseline adapter model to: {ADAPTER_SAVE_PATH_BASELINE}")
trainer.save_model(ADAPTER_SAVE_PATH_BASELINE)
print(f"✅ Baseline adapter and tokenizer saved to {ADAPTER_SAVE_PATH_BASELINE}")

# --- Finish W&B Run ---
if os.getenv('WANDB_PROJECT'): # Check if W&B was intended to be used
    print("📈 W&B run should be finished by the HuggingFace Trainer.")


print("\n🎉 Baseline model fine-tuning script with W&B integration finished successfully!")
print(f"➡️ Next steps: Check your W&B project for training logs. Then evaluate this model.")