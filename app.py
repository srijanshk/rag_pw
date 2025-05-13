import os
from typing import List
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import wandb 

from retriever import DenseRetriever
from pipeline import RAGPipeline
from evaluate import evaluate_pipeline

CHECKPOINTS_DIR = "checkpoints"
BEST_E2E_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "best_model_single_stage_e2e")
FINAL_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "final_model_E2E")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(BEST_E2E_MODEL_DIR, exist_ok=True)

TRAIN_FILE = "downloads/data/retriever/nq-train.json"
TEST_FILE = "downloads/data/retriever/nq-dev.json"
FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index" 
METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"

MAX_EPOCHS_E2E = 10      
PATIENCE_EPOCHS_E2E = 3   
SUBSET_EVAL_SIZE = 1500 


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
        positive_texts = [ctx.get("text", "") for ctx in pos_ctxs_sorted[:2] if ctx.get("text", "")]
        if not positive_texts:
            continue

        hard_negs = [ctx.get("text", "") for ctx in hard_neg_ctxs[:3] if ctx.get("text", "")]

        processed.append({
            "query": query,
            "positive_docs": positive_texts,
            "negative_docs": hard_negs,
            "answers": answers
        })
    return processed

def compute_answer_recall(rag_pipeline: RAGPipeline,
                          dataset: List[dict],
                          k: int) -> float:
    """
    Returns “Answer Recall@k”: the percentage of queries in the dataset
    for which at least one of the top-k retrieved passages
    contains the gold answer text.
    """
    hits = 0
    for item in tqdm(dataset, desc=f"Answer‐Recall@{k} eval", leave=False):
        query = item["query"]
        gold_answers = item["answers"]

        # Retrieve top-k docs via FAISS
        retrieved = rag_pipeline.dense_retriever.search(query, k)
        texts = [r["text"].lower() for r in retrieved]

        # Check if any retrieved doc contains any of the gold answers
        found = False
        for ans in gold_answers:
            ans_norm = ans.strip().lower()
            if not ans_norm:
                continue
            if any(ans_norm in doc_text for doc_text in texts):
                found = True
                break

        if found:
            hits += 1

    return hits / len(dataset) * 100.0


def main():
    batch_size_rag = 8


    retriever_model_name = "models/retriever_finetuned_e5_best"
    generator_model_name = "best_bart_model"

    # Load passage ID→row map to determine number of passages for memmap
    with open("/local00/student/shakya/id2row.json", "r", encoding="utf-8") as f:
        id2row_map = json.load(f)
    num_passages = len(id2row_map)

    top_k_retrieval = 10

    run_name = f"RAG_SingleStage_E2E_{MAX_EPOCHS_E2E}maxEp_P{PATIENCE_EPOCHS_E2E}"
    wandb.init(
        project="rag-qa-e2e-simplified", # New project or same, adjust as needed
        name=run_name,
        config={
            "batch_size_rag_e2e": batch_size_rag,
            "max_epochs_e2e": MAX_EPOCHS_E2E,
            "patience_e2e": PATIENCE_EPOCHS_E2E,
            "initial_retriever_model": retriever_model_name,
            "initial_generator_model": generator_model_name,
            "optimizer": "AdamW",
            "top_k_retrieval_inference": top_k_retrieval,
            "faiss_index_path": FAISS_INDEX_PATH,
            "metadata_path": METADATA_PATH,
            "subset_eval_size": SUBSET_EVAL_SIZE,
        },
        notes="Single-stage end-to-end RAG fine-tuning with early stopping based on subset evaluation.",
        resume="allow", id=wandb.util.generate_id()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print("Loading training data...")
    train_data = load_dpr_json(TRAIN_FILE)
    print("Loading test data...")
    full_test_data = load_dpr_json(TEST_FILE)

    subset_test_data_for_epochs = []
    if full_test_data:
        if len(full_test_data) > SUBSET_EVAL_SIZE:
            subset_test_data_for_epochs = full_test_data[:SUBSET_EVAL_SIZE] # Simple slice
            print(f"Created subset of {len(subset_test_data_for_epochs)} samples for inter-epoch evaluation.")
        else:
            subset_test_data_for_epochs = full_test_data
            print(f"Using all {len(subset_test_data_for_epochs)} test samples for inter-epoch evaluation (full set smaller than subset size).")
    else:
        print("No full test data loaded. Inter-epoch and final evaluations will be skipped.")

    if not train_data:
        print("No training data loaded. Exiting.")
        if wandb.run: wandb.finish()
        return

    print(f"Loaded {len(train_data)} training samples and {len(full_test_data)} full test samples.")



    print(f"\n🚀 Initializing Retriever with: {retriever_model_name}")
    retriever = DenseRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        device=device,
        model_name=retriever_model_name,
        passage_tokenizer_name_or_path=generator_model_name,
        id2row_path="/local00/student/shakya/id2row.json",
        input_ids_path="/local00/student/shakya/passage_input_ids.dat",
        attention_mask_path="/local00/student/shakya/passage_attention_mask.dat",
        num_passages=num_passages,
        passage_max_len=512,
        fine_tune=False,
        ef_search=1500,
        ef_construction=200,
    )
    print(f"\n🚀 Initializing RAG Pipeline with Generator: {generator_model_name}")
    steps_per_epoch = (len(train_data) + batch_size_rag - 1) // batch_size_rag
    total_scheduler_steps = steps_per_epoch * MAX_EPOCHS_E2E

    rag_pipeline_instance = RAGPipeline(
        model_name=generator_model_name,
        dense_retriever=retriever,
        device=device,
        train_generator=True,             
        train_retriever_end_to_end=True,
        total_steps_for_scheduler=total_scheduler_steps,
        k_retrieval_for_inference=top_k_retrieval,
        retriever_temperature_for_training=1.0
    )

    # --- Single Training Loop with Early Stopping ---
    avg_losses_per_epoch_e2e = []
    subset_eval_scores_e2e = []
    best_f1_on_subset_e2e = 0.0
    patience_counter_e2e = 0

    for epoch in range(MAX_EPOCHS_E2E):
        print(f"\n--- Training Epoch {epoch+1}/{MAX_EPOCHS_E2E} (End-to-End) ---")

        # RAGPipeline.train_pipeline internally loops for the number of epochs it's given.
        epoch_avg_loss_list = rag_pipeline_instance.train_pipeline(
            dataset=train_data,
            batch_size=batch_size_rag,
            epochs=1,
            stage_name=f"E2E_Train_Epoch{epoch+1}"
        )
        if epoch_avg_loss_list:
            avg_losses_per_epoch_e2e.append(epoch_avg_loss_list[0])

        current_epoch_subset_f1 = 0.0
        if subset_test_data_for_epochs:
            rag_pipeline_instance.generator.eval()
            if hasattr(rag_pipeline_instance.dense_retriever, 'query_encoder'):
                rag_pipeline_instance.dense_retriever.query_encoder.eval()

            print(f"Evaluating on SUBSET ({len(subset_test_data_for_epochs)} samples) after E2E Epoch {epoch+1}...")
            scores_subset = evaluate_pipeline(
                pipeline=rag_pipeline_instance,
                test_set=subset_test_data_for_epochs,
                verbose=False,
                log_path=os.path.join(CHECKPOINTS_DIR, f"e2e_subset_preds_epoch{epoch+1}.json"),
                top_k=top_k_retrieval,
                strategy="thorough",
            )
            subset_eval_scores_e2e.append(scores_subset)
            current_epoch_subset_f1 = scores_subset.get("F1", 0.0)

            answer_recall = compute_answer_recall(
                rag_pipeline_instance,
                subset_test_data_for_epochs,
                k=top_k_retrieval
            )

            print(f"Answer-Recall@{top_k_retrieval}: {answer_recall:.2f}%")
            if wandb.run:
                wandb.log({f"Answer_Recall@{top_k_retrieval}_Subset": answer_recall})



            print(f"Subset Eval (E2E Epoch {epoch+1}) — EM: {scores_subset.get('EM', 0.0):.2f}%, F1: {current_epoch_subset_f1:.2f}%")
            if wandb.run:
                wandb.log({
                    "EM_Subset_Eval_E2E": scores_subset.get("EM", 0.0),
                    "F1_Subset_Eval_E2E": current_epoch_subset_f1,
                    "Current_E2E_Epoch": epoch + 1
                })

            if current_epoch_subset_f1 > best_f1_on_subset_e2e:
                best_f1_on_subset_e2e = current_epoch_subset_f1
                patience_counter_e2e = 0
                print(f"✅ New best F1 on SUBSET: {best_f1_on_subset_e2e:.2f}%. Saving model to {BEST_E2E_MODEL_DIR}...")
                rag_pipeline_instance.generator.save_pretrained(os.path.join(BEST_E2E_MODEL_DIR, "generator"))
                rag_pipeline_instance.tokenizer.save_pretrained(os.path.join(BEST_E2E_MODEL_DIR, "generator"))
                if hasattr(rag_pipeline_instance.dense_retriever, 'save_query_encoder'):
                    rag_pipeline_instance.dense_retriever.save_query_encoder(os.path.join(BEST_E2E_MODEL_DIR, "retriever_query_encoder"))
            else:
                patience_counter_e2e += 1
                print(f"Subset F1 ({current_epoch_subset_f1:.2f}%) did not improve from best ({best_f1_on_subset_e2e:.2f}%). Patience: {patience_counter_e2e}/{PATIENCE_EPOCHS_E2E}.")

            if patience_counter_e2e >= PATIENCE_EPOCHS_E2E:
                print(f"Early stopping triggered after E2E Epoch {epoch+1} due to no improvement on subset evaluation.")
                break # Exit the training loop
        else:
            print("No subset test data available, skipping inter-epoch evaluation and early stopping for E2E training.")

    print("\n--- End of E2E Training Phase ---")


    # --- FINAL FULL EVALUATION ---
    print("\n🧪 Performing Final Full Evaluation on the entire test set...")
    rag_pipeline_instance.generator.eval()
    if hasattr(rag_pipeline_instance.dense_retriever, 'query_encoder'):
        rag_pipeline_instance.dense_retriever.query_encoder.eval()

    print(f"Evaluating model on {len(full_test_data)} FULL test samples...")
    final_full_scores = evaluate_pipeline(
        pipeline=rag_pipeline_instance,
        test_set=full_test_data,
        verbose=True,
        log_path=os.path.join(CHECKPOINTS_DIR, "final_e2e_FULL_predictions.json"),
        top_k=top_k_retrieval,
        strategy="thorough",
    )
    print("\n--- FINAL FULL Evaluation Scores ---")
    print(f"Exact Match (EM): {final_full_scores.get('EM', 0.0):.2f}%")
    print(f"F1 Score:         {final_full_scores.get('F1', 0.0):.2f}%")

    if wandb.run:
        wandb.log({
            "EM_Final_FULL_TestSet_E2E": final_full_scores.get("EM", 0.0),
            "F1_Final_FULL_TestSet_E2E": final_full_scores.get("F1", 0.0),
        })
    final_scores_path = os.path.join(CHECKPOINTS_DIR, "final_e2e_FULL_evaluation_scores.json")
    with open(final_scores_path, 'w') as f:
        json.dump(final_full_scores, f, indent=4)
    print(f"Final FULL evaluation scores saved to {final_scores_path}")

    if avg_losses_per_epoch_e2e:
        plt.figure(figsize=(10, 6))
        epochs_ran = range(1, len(avg_losses_per_epoch_e2e) + 1)
        plt.plot(epochs_ran, avg_losses_per_epoch_e2e, label='E2E Avg Training Loss per Epoch', marker='o')
        plt.xlabel("E2E Training Epoch")
        plt.ylabel("Average Loss")
        plt.title("E2E RAG Training Loss (Average per Epoch)")
        plt.legend()
        plt.xticks(epochs_ran)
        plt.grid(True)
        loss_plot_path = os.path.join(CHECKPOINTS_DIR, "training_loss_e2e_epochs.png")
        plt.savefig(loss_plot_path)
        if wandb.run: wandb.log({"training_loss_plot_e2e_epochs": wandb.Image(loss_plot_path)})
        print(f"E2E training loss plot saved to {loss_plot_path}")

    if subset_eval_scores_e2e:
        f1_scores_plot = [s.get("F1", 0.0) for s in subset_eval_scores_e2e]
        em_scores_plot = [s.get("EM", 0.0) for s in subset_eval_scores_e2e]
        epochs_plot = range(1, len(subset_eval_scores_e2e) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_plot, f1_scores_plot, label='F1 Score (E2E on Subset)', marker='o', color='green')
        plt.plot(epochs_plot, em_scores_plot, label='EM Score (E2E on Subset)', marker='x', color='blue')
        plt.xlabel("E2E Training Epoch")
        plt.ylabel("Score (%)")
        plt.title("E2E Evaluation Scores on Subset Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs_plot)
        eval_plot_path = os.path.join(CHECKPOINTS_DIR, "eval_scores_e2e_subset.png")
        plt.savefig(eval_plot_path)
        if wandb.run: wandb.log({"evaluation_scores_plot_e2e_subset": wandb.Image(eval_plot_path)})
        print(f"E2E evaluation scores plot (on subset) saved to {eval_plot_path}")

    if wandb.run:
        wandb.finish()
    print("\n--- Main script execution finished ---")
 


if __name__ == "__main__":
    main()