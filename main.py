import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import traceback
from tqdm.auto import tqdm
import json
import pandas as pd
import wandb 

from transformers import (
    AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer,
    get_scheduler, PreTrainedModel, AutoConfig
)
from torch.optim import AdamW

from NqDataset import NQDataset
from QuestionEncoder import QuestionEncoder
from DenseRetriever import DenseRetriever
from RagUtils import calculate_rag_loss, hybrid_retrieve_documents_for_batch, prepare_generator_inputs, retrieve_documents_for_batch
from utils import load_local_nq_json, custom_collate_fn, load_precomputed_sparse_results
from RagEval import evaluate_custom_rag_model

current_wandb_run = None # Global for wandb run object

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"Using device: {device}")

# --- Configuration ---
retriever_e5_model_name = "models/retriever_finetuned_e5_best"
generator_bart_model_name = "best_bart_model"
TRAIN_FILE = "downloads/data/gold_passages_info/nq_train.json"
TEST_FILE = "downloads/data/gold_passages_info/nq_dev.json"
SPARSE_TRAIN_FILE = "downloads/data/nq_train_sparse_retrieval.jsonl"
SPARSE_TEST_FILE = "downloads/data/nq_dev_sparse_retrieval.jsonl"
FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
K_SPARSE_PRECOMPUTED = 50
K_DENSE_RETRIEVAL = 10
n_docs_to_retrieve = 10
n_docs_to_retrieve_eval = 50
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
MODEL_SAVE_PATH = "./rag_train_hybrid_v4"
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "best_model")
MAX_QUESTION_LENGTH = 128 
MAX_ANSWER_LENGTH = 64   
MAX_COMBINED_LENGTH_FOR_GEN = 512 
TRAIN_DATA_LIMIT = None
EVAL_DATA_LIMIT = None
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_MAX_LOGGED_EXAMPLES = 100

# --- Main Function ---
def main():
    global current_wandb_run
    try:
        wandb.login()
        run = wandb.init(project="rag-pipeline-hybrid", config={
            "lr": LEARNING_RATE, "epochs": NUM_TRAIN_EPOCHS, "train_bs": TRAIN_BATCH_SIZE,
            "eval_bs": EVAL_BATCH_SIZE, "ret_model": retriever_e5_model_name, "gen_model": generator_bart_model_name,
            "max_q": MAX_QUESTION_LENGTH, "max_a": MAX_ANSWER_LENGTH, "max_comb_gen": MAX_COMBINED_LENGTH_FOR_GEN,
            "n_docs": n_docs_to_retrieve, "train_lim": TRAIN_DATA_LIMIT, "eval_lim": EVAL_DATA_LIMIT,
            "grad_accum": GRADIENT_ACCUMULATION_STEPS
        })
        current_wandb_run = run
        print("Weights & Biases initialized.")
    except Exception as e:
        print(f"Could not initialize wandb: {e}. Skipping wandb logging.")
        current_wandb_run = None

    # Initialize Models and Tokenizers
    print(f"Initializing DenseRetriever with E5: {retriever_e5_model_name}")
    dense_retriever = DenseRetriever(
        FAISS_INDEX_PATH, METADATA_PATH, device, retriever_e5_model_name,
        ef_search=1500, ef_construction=200, fine_tune=False) # fine_tune=False as QE is trained separately
    print("DenseRetriever initialized.")

    print(f"Loading E5 Question Encoder: {retriever_e5_model_name}")
    e5_config = AutoConfig.from_pretrained(retriever_e5_model_name)
    question_encoder = QuestionEncoder(config=e5_config, model_name_or_path=retriever_e5_model_name).to(device)
    e5_tokenizer = AutoTokenizer.from_pretrained(retriever_e5_model_name)
    print("E5 Question Encoder and Tokenizer loaded.")

    print(f"Loading BART Generator: {generator_bart_model_name}")
    generator = AutoModelForSeq2SeqLM.from_pretrained(generator_bart_model_name).to(device)
    bart_tokenizer = AutoTokenizer.from_pretrained(generator_bart_model_name)
    print(f"Generator max_position_embeddings: {generator.config.max_position_embeddings}")
    if hasattr(generator.config, "forced_bos_token_id") and generator.config.forced_bos_token_id is None and generator.config.bos_token_id is not None:
        generator.config.forced_bos_token_id = generator.config.bos_token_id
        print(f"Set generator.config.forced_bos_token_id to {generator.config.bos_token_id}")
    print("BART Generator and Tokenizer loaded.")

    print("Loading pre-computed sparse retrieval data...")
    try:
        train_sparse_data_lookup = load_precomputed_sparse_results(SPARSE_TRAIN_FILE)
        eval_sparse_data_lookup = load_precomputed_sparse_results(SPARSE_TEST_FILE)
    except Exception as e: return

    # Load Data
    print("Loading NQ dataset...")
    try:
        train_data_list = load_local_nq_json(TRAIN_FILE, limit=TRAIN_DATA_LIMIT)
        eval_data_list = load_local_nq_json(TEST_FILE, limit=EVAL_DATA_LIMIT)
    except Exception as e: return

    if not train_data_list: print("Training data list is empty. Exiting."); return
        
    train_dataset = NQDataset(train_data_list, train_sparse_data_lookup, e5_tokenizer, bart_tokenizer, MAX_QUESTION_LENGTH, MAX_ANSWER_LENGTH)
    eval_dataset = NQDataset(eval_data_list, eval_sparse_data_lookup, e5_tokenizer, bart_tokenizer, MAX_QUESTION_LENGTH, MAX_ANSWER_LENGTH)
    
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, collate_fn=custom_collate_fn if eval_data_list else None)

    # Optimizer and Scheduler
    optimizer = AdamW(list(question_encoder.parameters()) + list(generator.parameters()), lr=LEARNING_RATE)
    num_training_steps = NUM_TRAIN_EPOCHS * len(train_dataloader)
    lr_scheduler = None
    if num_training_steps > 0 :
        lr_scheduler = get_scheduler(
            "linear", optimizer, num_warmup_steps=int(0.1 * num_training_steps), # 10% warmup
            num_training_steps=num_training_steps)
        print(f"Optimizer and LR scheduler initialized for {num_training_steps} steps.")
    elif NUM_TRAIN_EPOCHS > 0:
         print("Warning: Training dataloader is empty. Check data loading and TRAIN_DATA_LIMIT."); return
    else: print("No training steps. Skipping training.")

    best_f1_score = 0.0 

    # Training Loop
    if num_training_steps > 0:
        print("\n--- Starting End-to-End RAG Training ---")
        total_steps_trained = 0 # To track total optimizer steps for wandb logging
        for epoch in range(NUM_TRAIN_EPOCHS):
            question_encoder.train()
            generator.train()
            epoch_train_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")

            for batch_idx, batch in enumerate(progress_bar):
                q_input_ids = batch["input_ids"].to(device)
                q_attention_mask = batch["attention_mask"].to(device)
                target_labels = batch["labels"].to(device)
                original_qs_str = batch["original_question"]
                batch_precomputed_sparse = batch["precomputed_sparse_docs"] 

                try:
                    query_embeddings = question_encoder(q_input_ids, q_attention_mask)[0]
                    # retrieved = retrieve_documents_for_batch(query_embeddings, dense_retriever, n_docs_to_retrieve, True)
                    # Perform hybrid retrieval
                    hybrid_retrieved_info = hybrid_retrieve_documents_for_batch(
                        query_embeddings_batch=query_embeddings,
                        batch_precomputed_sparse_docs=batch_precomputed_sparse,
                        dense_retriever=dense_retriever, # DenseRetriever instance
                        final_k=n_docs_to_retrieve, # Or FINAL_K_FOR_GENERATOR
                        k_dense_to_fetch=K_DENSE_RETRIEVAL, # How many dense docs to fetch before fusion
                        # fusion_k_constant can be a default in the function or passed
                        device=device 
                    )

                    gen_inputs = prepare_generator_inputs(original_qs_str, hybrid_retrieved_info["retrieved_doc_titles"],
                                                          hybrid_retrieved_info["retrieved_doc_texts"], bart_tokenizer,
                                                          MAX_COMBINED_LENGTH_FOR_GEN, device)
                    
                    loss = calculate_rag_loss(query_embeddings, hybrid_retrieved_info["retrieved_doc_embeddings"],
                                              gen_inputs["generator_input_ids"], gen_inputs["generator_attention_mask"],
                                              target_labels, generator, bart_tokenizer.pad_token_id,
                                              n_docs_to_retrieve, device)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss batch {batch_idx}. Skipping."); optimizer.zero_grad(); continue
                    
                    loss_accum = loss / GRADIENT_ACCUMULATION_STEPS
                    loss_accum.backward()

                    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or \
                       (batch_idx + 1) == len(train_dataloader):
                        torch.nn.utils.clip_grad_norm_(list(question_encoder.parameters()) + list(generator.parameters()), 1.0)
                        optimizer.step()
                        if lr_scheduler: lr_scheduler.step()
                        optimizer.zero_grad()
                        total_steps_trained +=1 
                    
                    epoch_train_loss += loss.item() # Log the non-accumulated loss for interpretability
                    current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler and lr_scheduler.get_last_lr() else LEARNING_RATE
                    progress_bar.set_postfix({'loss': loss.item(), 'lr': current_lr})
                    
                    if current_wandb_run: 
                        wandb.log({"step_train_loss": loss.item(), "learning_rate": current_lr, "step": total_steps_trained * GRADIENT_ACCUMULATION_STEPS + batch_idx}) # More granular step
                
                except Exception as e:
                    print(f"Error during training step {batch_idx} in epoch {epoch+1}: {e}")
                    traceback.print_exc()
                    print("Exiting training due to error."); 
                    if current_wandb_run: wandb.finish(exit_code=1)
                    return 

            avg_epoch_loss = epoch_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else float('nan')
            print(f"Avg Train Loss Epoch {epoch+1}: {avg_epoch_loss}")
            log_data_epoch = {"epoch_train_loss": avg_epoch_loss, "epoch": epoch + 1}
            
            if current_wandb_run: wandb.log(log_data_epoch)

            if eval_dataloader and len(eval_dataloader) > 0:
                eval_metrics, logged_samples = evaluate_custom_rag_model(
                    question_encoder, dense_retriever, generator, eval_dataloader,
                    e5_tokenizer, bart_tokenizer, 
                    n_docs_to_retrieve_eval,
                    MAX_COMBINED_LENGTH_FOR_GEN, MAX_ANSWER_LENGTH, device, str(epoch+1),
                    EVAL_MAX_LOGGED_EXAMPLES, 50, current_wandb_run
                )
                log_data_epoch.update(eval_metrics)

                if current_wandb_run and logged_samples:
                    wandb.log({**eval_metrics, "epoch": epoch + 1})
                    if logged_samples:
                        try:
                            df_samples = pd.DataFrame(logged_samples)
                            wandb.log({"evaluation_examples_epoch_" + str(epoch+1): wandb.Table(dataframe=df_samples)})
                        except Exception as df_e: print(f"Wandb table log error: {df_e}")
                
                current_f1 = eval_metrics.get("f1", 0.0)
                if current_f1 > best_f1_score:
                    best_f1_score = current_f1
                    print(f"New best F1 score: {best_f1_score:.4f} at epoch {epoch+1}. Saving best model...")
                    if not os.path.exists(BEST_MODEL_SAVE_PATH):
                        os.makedirs(BEST_MODEL_SAVE_PATH)
                    question_encoder.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "question_encoder"))
                    generator.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "generator"))
                    e5_tokenizer.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "question_encoder"))
                    e5_tokenizer.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "question_tokenizer"))
                    bart_tokenizer.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "generator_tokenizer"))
                    generator.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "bart_generator"))
                    bart_tokenizer.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "bart_generator"))
                    
                    with open(os.path.join(BEST_MODEL_SAVE_PATH, "best_score.json"), "w") as f:
                        json.dump({"best_f1_score": best_f1_score, "epoch": epoch+1}, f)
                    print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")
        
        print(f"\nTraining done. Saving to {MODEL_SAVE_PATH}")
        if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
        question_encoder.save_pretrained(os.path.join(MODEL_SAVE_PATH, "question_encoder"))
        generator.save_pretrained(os.path.join(MODEL_SAVE_PATH, "generator"))
        e5_tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, "question_encoder"))
        e5_tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, "question_tokenizer"))
        bart_tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, "generator_tokenizer"))
        generator.save_pretrained(os.path.join(MODEL_SAVE_PATH, "bart_generator"))
        bart_tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, "bart_generator"))
        print(f"RAG components saved to {MODEL_SAVE_PATH}")

    if current_wandb_run: wandb.finish()

if __name__ == "__main__":
    main()
