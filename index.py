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
from RagUtils import calculate_rag_loss, prepare_generator_inputs, retrieve_documents_for_batch
from utils import load_local_nq_json, custom_collate_fn
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
FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
n_docs_to_retrieve = 5
n_docs_to_retrieve_eval = 10
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
MODEL_SAVE_PATH = "./rag_custom_trained_final_v3"
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "best_model")
MAX_QUESTION_LENGTH = 128 
MAX_ANSWER_LENGTH = 64   
MAX_COMBINED_LENGTH_FOR_GEN = 512 
TRAIN_DATA_LIMIT = None
EVAL_DATA_LIMIT = None
GRADIENT_ACCUMULATION_STEPS = 6
EVAL_MAX_LOGGED_EXAMPLES = 3

# --- Main Function ---
def main():
    global current_wandb_run
    try:
        wandb.login()
        run = wandb.init(project="rag-pipeline", config={
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

    # Load Data
    print("Loading NQ dataset...")
    try:
        train_data_list = load_local_nq_json(TRAIN_FILE, limit=TRAIN_DATA_LIMIT)
        eval_data_list = load_local_nq_json(TEST_FILE, limit=EVAL_DATA_LIMIT)
    except Exception as e: return

    if not train_data_list: print("Training data list is empty. Exiting."); return
        
    train_dataset = NQDataset(train_data_list, e5_tokenizer, bart_tokenizer, MAX_QUESTION_LENGTH, MAX_ANSWER_LENGTH)
    eval_dataset = NQDataset(eval_data_list, e5_tokenizer, bart_tokenizer, MAX_QUESTION_LENGTH, MAX_ANSWER_LENGTH)
    
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

                try:
                    query_embeddings = question_encoder(q_input_ids, q_attention_mask)[0]
                    retrieved = retrieve_documents_for_batch(query_embeddings, dense_retriever, n_docs_to_retrieve, True)
                    gen_inputs = prepare_generator_inputs(original_qs_str, retrieved["retrieved_doc_titles"],
                                                          retrieved["retrieved_doc_texts"], bart_tokenizer,
                                                          MAX_COMBINED_LENGTH_FOR_GEN, device)
                    
                    loss = calculate_rag_loss(query_embeddings, retrieved["retrieved_doc_embeddings"],
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
                    EVAL_MAX_LOGGED_EXAMPLES, current_wandb_run
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
                    e5_tokenizer.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "question_tokenizer"))
                    bart_tokenizer.save_pretrained(os.path.join(BEST_MODEL_SAVE_PATH, "generator_tokenizer"))
                    # Save any other relevant info, like best F1 score itself
                    with open(os.path.join(BEST_MODEL_SAVE_PATH, "best_score.json"), "w") as f:
                        json.dump({"best_f1_score": best_f1_score, "epoch": epoch+1}, f)
                    print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")
        
        print(f"\nTraining done. Saving to {MODEL_SAVE_PATH}")
        if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
        question_encoder.save_pretrained(os.path.join(MODEL_SAVE_PATH, "question_encoder"))
        generator.save_pretrained(os.path.join(MODEL_SAVE_PATH, "generator"))
        e5_tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, "question_tokenizer"))
        bart_tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, "generator_tokenizer"))
        print(f"RAG components saved to {MODEL_SAVE_PATH}")

    # Inference Example
    print("\n--- Custom RAG Inference Example ---")
    question_encoder.eval(); generator.eval()
    test_question = "What is the Albert Einstein College of Medicine known for?"
    print(f"Test Question: {test_question}")
    q_tok_inf = e5_tokenizer(test_question, return_tensors="pt", max_length=MAX_QUESTION_LENGTH, truncation=True).to(device)
    
    with torch.no_grad():
        q_embed_inf = question_encoder(q_tok_inf.input_ids, q_tok_inf.attention_mask)[0]
        ret_inf = retrieve_documents_for_batch(q_embed_inf, dense_retriever, n_docs_to_retrieve, True)
        gen_in_inf = prepare_generator_inputs([test_question], ret_inf["retrieved_doc_titles"],
                                              ret_inf["retrieved_doc_texts"], bart_tokenizer,
                                              MAX_COMBINED_LENGTH_FOR_GEN, device)
        
        if gen_in_inf["generator_input_ids"].numel() > 0:
            # For inference, select based on doc scores (similar to eval)
            # This part can be enhanced for better answer selection from n_docs
            query_embeddings_inf = q_embed_inf
            retrieved_doc_embeddings_inf = ret_inf["retrieved_doc_embeddings"].to(device)
            
            expanded_query_embeddings_inf = query_embeddings_inf.unsqueeze(1)
            doc_scores_inf = torch.bmm(expanded_query_embeddings_inf, retrieved_doc_embeddings_inf.transpose(1, 2)).squeeze(1)
            best_doc_idx_inf = torch.argmax(doc_scores_inf, dim=1)[0].item() # Assuming batch size 1 for this simple inference

            input_ids_for_gen = gen_in_inf["generator_input_ids"][best_doc_idx_inf].unsqueeze(0)
            attention_mask_for_gen = gen_in_inf["generator_attention_mask"][best_doc_idx_inf].unsqueeze(0)

            generated_ids_inf = generator.generate(
                input_ids=input_ids_for_gen, 
                attention_mask=attention_mask_for_gen,
                num_beams=4, max_length=MAX_ANSWER_LENGTH + 20, early_stopping=True,
                pad_token_id=bart_tokenizer.eos_token_id if bart_tokenizer.eos_token_id is not None else bart_tokenizer.pad_token_id,
                eos_token_id=bart_tokenizer.eos_token_id,
                decoder_start_token_id=generator.config.decoder_start_token_id
            )
            gen_text_inf = bart_tokenizer.batch_decode(generated_ids_inf, skip_special_tokens=True)
            print(f"Generated Answer: {gen_text_inf}")
            print(f"Based on document title: {ret_inf['retrieved_doc_titles'][0][best_doc_idx_inf]}")
        else:
            print("No context generated for inference.")

    if current_wandb_run: wandb.finish()

if __name__ == "__main__":
    main()