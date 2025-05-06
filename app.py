import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # Or your desired GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # If needed

import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import wandb 

from retriever import DenseRetriever
from pipeline import RAGPipeline
from evaluate import evaluate_pipeline # From your evaluate.py

CHECKPOINTS_DIR = "checkpoints"
BEST_MODEL_DIR_STAGE3 = os.path.join(CHECKPOINTS_DIR, "best_model_stage3")
FINAL_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "final_model_E2E")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR_STAGE3, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

TRAIN_FILE = "downloads/data/retriever/nq-train.json"
TEST_FILE = "downloads/data/retriever/nq-dev.json"
FAISS_INDEX_PATH = "wikipedia_faiss_index" 
METADATA_PATH = "wikipedia_metadata.jsonl"


def load_dpr_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}")
        return []

    processed = []
    for item in data:
        query = item.get("question")
        answers = item.get("answers", [])
        pos_ctxs = item.get("positive_ctxs", [])
        hard_neg_ctxs = item.get("hard_negative_ctxs", [])

        if not query or not pos_ctxs:
            continue
        
        pos_ctxs_sorted = sorted(pos_ctxs, key=lambda x: x.get("score", 0), reverse=True)
        top_pos = pos_ctxs_sorted[0]
        top_text = top_pos.get("text", "")

        if not top_text: 
            continue

        weak_positives_as_negs = [
            ctx.get("text", "") for ctx in pos_ctxs_sorted[-2:]
            if ctx.get("text", "") and ctx.get("text", "") != top_text
        ]
        hard_negs = [ctx.get("text", "") for ctx in hard_neg_ctxs[:2] if ctx.get("text", "")]
        combined_negs = [neg for neg in (weak_positives_as_negs + hard_negs) if neg]

        processed.append({
            "query": query,
            "positive_docs": [top_text], 
            "negative_docs": combined_negs, 
            "answers": answers
        })
    return processed

def main():
    batch_size_retriever_ft = 8
    batch_size_rag = 16
    
    stage1_epochs = 0
    stage2_epochs = 2
    stage3_epochs = 6

    retriever_model_name = "intfloat/e5-large-v2"
    generator_model_name = "facebook/bart-base"
    
    retriever_lr_config = 2e-5 
    generator_lr_config = 3e-5 
    
    top_k_retrieval = 50

    wandb.init(
        project="rag-qa-e2e", 
        name="RAG_E2E_With_Stages_v3_Fix", # Changed name to reflect fix
        config={
            "batch_size_retriever_ft": batch_size_retriever_ft,
            "batch_size_rag": batch_size_rag,
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "stage3_epochs": stage3_epochs,
            "retriever_model": retriever_model_name,
            "generator_model": generator_model_name,
            "optimizer": "AdamW",
            "retriever_lr_config": retriever_lr_config,
            "generator_lr_config": generator_lr_config,
            "top_k_retrieval_inference": top_k_retrieval,
            "faiss_index_path": FAISS_INDEX_PATH,
            "metadata_path": METADATA_PATH,
        },
        notes="Three-stage fine-tuning: retriever (optional), generator, end-to-end RAG. Corrected wandb.watch.",
        resume="allow", id=wandb.util.generate_id()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print("Loading training data...")
    train_data = load_dpr_json(TRAIN_FILE)
    print("Loading test data...")
    test_data = load_dpr_json(TEST_FILE)
    
    if not train_data:
        print("No training data loaded. Exiting.")
        wandb.finish()
        return
    if not test_data:
        print("No test data loaded. Evaluation will be skipped if Stage 3 runs.")

    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")

    retriever_path_stage1 = os.path.join(CHECKPOINTS_DIR, "retriever_stage1_query_encoder")
    
    if stage1_epochs > 0:
        print("\n🔹 Stage 1: Retriever Fine-tuning (Standalone)")
        retriever_stage1 = DenseRetriever(
            index_path=FAISS_INDEX_PATH,
            metadata_path=METADATA_PATH,
            device=device,
            model_name=retriever_model_name,
            fine_tune_flag=True
        )
        if hasattr(retriever_stage1, 'query_encoder'):
             wandb.watch(retriever_stage1.query_encoder, log="all", log_freq=100) # Removed 'name'

        retriever_train_data_formatted = [
            {
                "query": item["query"],
                "positive_doc": item["positive_docs"][0] if item["positive_docs"] else None,
                "negative_docs": item.get("negative_docs", [])
            }
            for item in train_data if item.get("positive_docs")
        ]
        if not retriever_train_data_formatted:
            print("Warning: No data suitable for retriever fine-tuning (Stage 1) after formatting.")
        else:
            print(f"Starting Stage 1 training with {len(retriever_train_data_formatted)} formatted samples.")
            for epoch in range(stage1_epochs):
                print(f"\nRetriever Standalone Epoch {epoch+1}/{stage1_epochs}")
                np.random.shuffle(retriever_train_data_formatted)
                num_batches = (len(retriever_train_data_formatted) + batch_size_retriever_ft - 1) // batch_size_retriever_ft
                batch_indices = range(0, len(retriever_train_data_formatted), batch_size_retriever_ft)
                with tqdm(total=len(batch_indices), desc=f"Retriever FT Epoch {epoch+1}") as pbar:
                    for i_idx, i_start in enumerate(batch_indices):
                        batch = retriever_train_data_formatted[i_start : i_start + batch_size_retriever_ft]
                        if not batch: continue
                        loss = retriever_stage1.fine_tune_on_batch(batch)
                        pbar.update(1)
                        if loss is not None:
                            wandb.log({"retriever_standalone_loss": loss, 
                                       "stage1_epoch": epoch+1, 
                                       "stage1_batch_idx": i_idx})
            retriever_stage1.save_query_encoder(retriever_path_stage1)
            print(f"Retriever fine-tuning (Stage 1) complete. Saved to {retriever_path_stage1}")
            retriever = DenseRetriever.load_from_paths(
                model_load_path=retriever_path_stage1,
                index_path=FAISS_INDEX_PATH,
                metadata_path=METADATA_PATH,
                device=device,
                base_model_name_for_doc_encoder=retriever_model_name
            )
    else:
        print("\nSkipping Stage 1: Retriever Fine-tuning (Standalone)")
        if os.path.exists(retriever_path_stage1) and os.listdir(retriever_path_stage1): # Check if dir exists and is not empty
            print(f"Loading retriever from pre-trained Stage 1 checkpoint: {retriever_path_stage1}")
            retriever = DenseRetriever.load_from_paths(
                model_load_path=retriever_path_stage1,
                index_path=FAISS_INDEX_PATH,
                metadata_path=METADATA_PATH,
                device=device,
                base_model_name_for_doc_encoder=retriever_model_name
            )
        else:
            print(f"Initializing fresh retriever for subsequent stages (no Stage 1 checkpoint at {retriever_path_stage1}).")
            retriever = DenseRetriever(
                index_path=FAISS_INDEX_PATH,
                metadata_path=METADATA_PATH,
                device=device,
                model_name=retriever_model_name,
                fine_tune_flag=False
            )

    # --- Stage 2: Generator-only fine-tuning ---
    generator_path_stage2 = os.path.join(CHECKPOINTS_DIR, "generator_stage2")
    rag_pipeline_instance = RAGPipeline(
            dense_retriever=retriever,
            device=device,
            train_generator=False, 
            train_retriever_end_to_end=False, 
            total_steps_for_scheduler=1, # Placeholder
            k_retrieval_for_inference=top_k_retrieval
        )

    if stage2_epochs > 0:
        print("\n🔄 Stage 2: Generator-only fine-tuning")
        retriever.disable_fine_tuning_mode()
        
        steps_for_stage2 = (len(train_data) + batch_size_rag - 1) // batch_size_rag * stage2_epochs
        rag_pipeline_instance = RAGPipeline( # Overwrite previous placeholder instance
            dense_retriever=retriever,
            device=device,
            train_generator=True,
            train_retriever_end_to_end=False,
            total_steps_for_scheduler=steps_for_stage2,
            k_retrieval_for_inference=top_k_retrieval,
            retriever_temperature_for_training=1.0
        )
        wandb.watch(rag_pipeline_instance.generator, log="all", log_freq=100) # Removed 'name'

        print("RAG pipeline initialized for Stage 2 (Generator Only).")
        rag_pipeline_instance.train_pipeline(
            train_data, 
            batch_size=batch_size_rag, 
            epochs=stage2_epochs, 
            stage_name="Stage2_GeneratorOnly"
        )
        rag_pipeline_instance.generator.save_pretrained(generator_path_stage2)
        rag_pipeline_instance.tokenizer.save_pretrained(generator_path_stage2)
        print(f"Generator fine-tuning (Stage 2) complete. Saved to {generator_path_stage2}")


    # --- Stage 3: End-to-End RAG Fine-tuning ---
    all_stage3_losses = []
    eval_scores_stage3 = []
    best_f1_stage3 = 0.0

    if stage3_epochs > 0:
        print("\n🔄 Stage 3: End-to-End RAG fine-tuning")
        
        rag_pipeline_instance.train_generator = True # Ensure generator is trained
        rag_pipeline_instance.train_retriever_end_to_end = True # Enable E2E retriever training
        
        steps_for_stage3 = (len(train_data) + batch_size_rag - 1) // batch_size_rag * stage3_epochs
        rag_pipeline_instance._init_optimizers_and_scheduler( # Re-initialize optimizer and scheduler for E2E
            num_training_steps_for_current_stage=steps_for_stage3
        ) 

        wandb.watch(rag_pipeline_instance.generator, log="all", log_freq=100) # Removed 'name'
        if hasattr(rag_pipeline_instance.dense_retriever, 'query_encoder'):
             wandb.watch(rag_pipeline_instance.dense_retriever.query_encoder, log="all", log_freq=100) # Removed 'name'

        print("RAG pipeline re-configured for Stage 3 (End-to-End).")
        
        for epoch in range(stage3_epochs):
            print(f"\nStage 3 Epoch {epoch+1}/{stage3_epochs} (End-to-End)")
            epoch_loss_list = rag_pipeline_instance.train_pipeline(
                train_data, 
                batch_size=batch_size_rag, 
                epochs=1, 
                stage_name=f"Stage3_E2E_Epoch{epoch+1}"
            )
            if epoch_loss_list:
                 all_stage3_losses.extend(epoch_loss_list)

            if test_data:
                rag_pipeline_instance.generator.eval()
                if hasattr(rag_pipeline_instance.dense_retriever, 'query_encoder'):
                    rag_pipeline_instance.dense_retriever.query_encoder.eval()
                
                print(f"Evaluating Stage 3, Epoch {epoch+1}...")
                scores = evaluate_pipeline(
                    rag_pipeline_instance, 
                    test_data, 
                    verbose=False, 
                    log_path=os.path.join(CHECKPOINTS_DIR, f"stage3_predictions_epoch{epoch+1}.json"), 
                    top_k=top_k_retrieval
                )
                eval_scores_stage3.append(scores)
                
                print(f"Eval (Epoch {epoch+1}) — EM: {scores['EM']:.2f}%, F1: {scores['F1']:.2f}%")
                wandb.log({
                    "EM_Stage3": scores["EM"],
                    "F1_Stage3": scores["F1"],
                    "epoch_stage3": epoch + 1
                })

                if scores["F1"] > best_f1_stage3:
                    best_f1_stage3 = scores["F1"]
                    print(f"✅ New best F1 in Stage 3: {best_f1_stage3:.2f}%. Saving model to {BEST_MODEL_DIR_STAGE3}...")
                    rag_pipeline_instance.generator.save_pretrained(os.path.join(BEST_MODEL_DIR_STAGE3, "generator"))
                    rag_pipeline_instance.tokenizer.save_pretrained(os.path.join(BEST_MODEL_DIR_STAGE3, "tokenizer"))
                    if hasattr(rag_pipeline_instance.dense_retriever, 'save_query_encoder'):
                        rag_pipeline_instance.dense_retriever.save_query_encoder(os.path.join(BEST_MODEL_DIR_STAGE3, "retriever_query_encoder"))
            else:
                print("No test data loaded, skipping evaluation for Stage 3.")
        print("End-to-End fine-tuning (Stage 3) complete.")
    else:
        print("\nSkipping Stage 3: End-to-End RAG fine-tuning")

    print(f"\n💾 Saving final model components to '{FINAL_MODEL_DIR}'...")
    if hasattr(rag_pipeline_instance, 'generator') and rag_pipeline_instance.generator is not None:
        rag_pipeline_instance.generator.save_pretrained(os.path.join(FINAL_MODEL_DIR, "generator"))
        rag_pipeline_instance.tokenizer.save_pretrained(os.path.join(FINAL_MODEL_DIR, "tokenizer"))
    else:
        print("Generator not available for final saving (possibly skipped all training stages).")

    if hasattr(rag_pipeline_instance, 'dense_retriever') and \
       hasattr(rag_pipeline_instance.dense_retriever, 'save_query_encoder') and \
       hasattr(rag_pipeline_instance.dense_retriever, 'query_encoder') and \
       rag_pipeline_instance.dense_retriever.query_encoder is not None:
        rag_pipeline_instance.dense_retriever.save_query_encoder(os.path.join(FINAL_MODEL_DIR, "retriever_query_encoder"))
    else:
        print("Retriever query encoder not available for final saving.")
    print("Final model components potentially saved.")


    if all_stage3_losses: 
        plt.figure(figsize=(10, 6))
        plt.plot(all_stage3_losses, label='Stage 3 Training Loss (Avg per Batch)')
        plt.xlabel("Training Batch (Cumulative in Stage 3)")
        plt.ylabel("Loss")
        plt.title("Stage 3 RAG End-to-End Training Loss")
        plt.legend()
        loss_plot_path = os.path.join(CHECKPOINTS_DIR, "training_loss_stage3.png")
        plt.savefig(loss_plot_path)
        wandb.log({"training_loss_plot_stage3": wandb.Image(loss_plot_path)})
        print(f"Training loss plot saved to {loss_plot_path}")

    if eval_scores_stage3:
        f1_scores_plot = [s["F1"] for s in eval_scores_stage3]
        em_scores_plot = [s["EM"] for s in eval_scores_stage3]
        epochs_plot = range(1, len(eval_scores_stage3) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_plot, f1_scores_plot, label='F1 Score (Stage 3)', marker='o', color='green')
        plt.plot(epochs_plot, em_scores_plot, label='EM Score (Stage 3)', marker='x', color='blue')
        plt.xlabel("Epoch (Stage 3)")
        plt.ylabel("Score (%)")
        plt.title("Evaluation Scores Over Epochs (Stage 3)")
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs_plot)
        eval_plot_path = os.path.join(CHECKPOINTS_DIR, "eval_scores_stage3.png")
        plt.savefig(eval_plot_path)
        wandb.log({"evaluation_scores_plot_stage3": wandb.Image(eval_plot_path)})
        print(f"Evaluation scores plot saved to {eval_plot_path}")

    wandb.finish()
    print("\n--- Main script execution finished ---")

if __name__ == "__main__":
    main()