import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" # Or "expandable_segments:True"

import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import wandb
import logging

from xapian_retriever import XapianRetriever
from sparse_rag_pipeline import SparseRAGPipeline
from _evaluate import evaluate_pipeline, evaluate_pipeline_sparse
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW


# --- Configuration ---
CHECKPOINTS_DIR = "checkpoints_sparse_rag_optionA_TB" # TB for TrueBatch
BEST_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "best_model")
FINAL_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "final_model")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# Data and Model Paths
# THESE NOW POINT TO THE OUTPUTS OF offline_retrieve_contexts.py
TRAIN_FILE_PRE_RETRIEVED = "downloads/data/xapian/nq-train_xapian_contexts.jsonl"
DEV_FILE_PRE_RETRIEVED = "downloads/data/xapian/nq-dev_xapian_contexts.jsonl"

XAPIAN_DB_PATH = "/local00/student/shakya/wikipedia_xapian_db"

GENERATOR_MODEL_NAME = "best_bart_model"

# Training Hyperparameters
BATCH_SIZE = 8 
EPOCHS = 8      # Number of training epochs
GENERATOR_LR = 2e-5 # Adjusted LR
K_MARGINALIZATION_TRAINING = 10 # N: Number of pre-retrieved docs to use from the .jsonl file for one sample
K_RETRIEVAL_INFERENCE = 50    # K: Number of docs for live Xapian retrieval during inference/eval
WEIGHT_DECAY = 0.01

# Option A specific params
USE_XAPIAN_SCORES_WEIGHTS = True
LOSS_WEIGHTING_TEMP = 0.1 # Temperature for softmaxing Xapian scores in loss
EVAL_WITH_PRE_RETRIEVED_CONTEXTS = True # Flag to use pre-retrieved contexts for evaluation

# Setup Logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) # Main app logger


def load_jsonl_with_pre_retrieved_contexts(file_path):
    """Loads data from a JSONL file where each line is a JSON object."""
    data = []
    logger.info(f"Attempting to load pre-retrieved data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    # Basic validation for required fields from offline script
                    if "query" in item and "answers" in item and "retrieved_contexts" in item:
                        data.append(item)
                    else:
                        logger.warning(f"Skipping malformed item on line {line_num+1} in {file_path}: missing essential keys.")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON on line {line_num+1} in {file_path}: {line.strip()}")
        logger.info(f"Successfully loaded {len(data)} samples from {file_path}")
    except FileNotFoundError:
        logger.error(f"FATAL: Pre-retrieved data file not found: {file_path}")
        return [] # Return empty list on critical error
    return data


def main_app_sparse_option_a():
    # --- WandB Initialization ---
    # Ensure WANDB_API_KEY is set in your environment if needed
    try:
        wandb.init(
            project="sparse-rag-optionA-TB", # New project name
            name="SparseRAG_Xapian_OptA_TrueBatch_NQ",
            config={
                "batch_size_original_samples": BATCH_SIZE,
                "epochs": EPOCHS,
                "generator_model": GENERATOR_MODEL_NAME,
                "optimizer": "AdamW",
                "generator_lr": GENERATOR_LR,
                "weight_decay": WEIGHT_DECAY,
                "k_marginalization_training": K_MARGINALIZATION_TRAINING,
                "k_retrieval_inference": K_RETRIEVAL_INFERENCE,
                "use_xapian_scores_for_loss_weights": USE_XAPIAN_SCORES_WEIGHTS,
                "loss_weighting_temperature": LOSS_WEIGHTING_TEMP,
                "xapian_db_path_for_inference": XAPIAN_DB_PATH,
                "train_file_pre_retrieved": TRAIN_FILE_PRE_RETRIEVED,
                "dev_file_pre_retrieved": DEV_FILE_PRE_RETRIEVED,
            },
            resume="allow",
            id=wandb.util.generate_id()
        )
        logger.info("WandB initialized successfully.")
    except Exception as e:
        logger.error(f"WandB initialization failed: {e}. Training will continue without WandB logging.", exc_info=True)


    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        # Potentially clear cache if memory issues persist from previous runs
        # torch.cuda.empty_cache()


    # --- Load Pre-Retrieved Data ---
    logger.info(f"Loading pre-retrieved training data from: {TRAIN_FILE_PRE_RETRIEVED}")
    train_data = load_jsonl_with_pre_retrieved_contexts(TRAIN_FILE_PRE_RETRIEVED)
    logger.info(f"Loading pre-retrieved development/test data from: {DEV_FILE_PRE_RETRIEVED}")
    # For evaluation, we often use live retrieval, but if dev contexts are pre-fetched, load them.
    # If not, load the original dev file for queries/answers and let pipeline do live retrieval.
    # For this example, assume dev contexts are also pre-fetched for consistency in data format.
    dev_data = load_jsonl_with_pre_retrieved_contexts(DEV_FILE_PRE_RETRIEVED)


    if not train_data:
        logger.error("No training data loaded. Please ensure the offline retrieval script has run successfully. Exiting.")
        if wandb.run: wandb.finish(exit_code=1)
        return

    logger.info(f"Loaded {len(train_data)} pre-retrieved training samples.")
    if dev_data:
        logger.info(f"Loaded {len(dev_data)} pre-retrieved dev/test samples.")
    else:
        logger.warning("No development/test data loaded or it failed to load. Evaluation will be skipped.")


    # --- Initialize Retriever (for inference/evaluation only) ---
    logger.info(f"Initializing XapianRetriever for inference with DB: {XAPIAN_DB_PATH}")
    try:
        # This retriever is passed to the pipeline for its .generate_answer() method
        inference_xapian_retriever = XapianRetriever(db_path=XAPIAN_DB_PATH, use_bm25=True)
        # Quick test
        sample_query_for_test = train_data[0]['query'] if train_data else "test query for xapian"
        test_search_results = inference_xapian_retriever.search(sample_query_for_test, k=1)
        if test_search_results:
            logger.info(f"Inference Xapian retriever test successful. Retrieved: {test_search_results[0][0].get('title', 'N/A')}")
        else:
            logger.warning("Inference Xapian retriever test returned no results. Check DB path and content if using live eval.")
    except Exception as e:
        logger.error(f"Failed to initialize inference XapianRetriever: {e}", exc_info=True)
        if wandb.run: wandb.finish(exit_code=1)
        return


    # Calculate total steps for scheduler based on actual training data and epochs
    num_batches_per_epoch_calc = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE
    total_scheduler_steps = num_batches_per_epoch_calc * EPOCHS

    # Create the pipeline instance
    # Pass all necessary initial_hyperparameters to load_model if you implement loading
    initial_hyperparameters_for_pipeline = {
        'k_retrieval_for_training_marginalization': K_MARGINALIZATION_TRAINING,
        'k_retrieval_for_inference': K_RETRIEVAL_INFERENCE,
        'generator_lr': GENERATOR_LR,
        'total_steps_for_scheduler': total_scheduler_steps,
        'max_context_length': 512, 
        'max_answer_length': 128, 
        'use_xapian_scores_for_loss_weights': USE_XAPIAN_SCORES_WEIGHTS,
        'loss_weighting_temperature': LOSS_WEIGHTING_TEMP
    }

    sparse_rag_pipeline = SparseRAGPipeline(
        sparse_retriever=inference_xapian_retriever, # For generate_answer
        device=device,
        generator_model_name=GENERATOR_MODEL_NAME,
        **initial_hyperparameters_for_pipeline
    )
    # If you add AdamW optimizer options like weight_decay:
    sparse_rag_pipeline.optimizer = AdamW(sparse_rag_pipeline.generator.parameters(), lr=GENERATOR_LR, weight_decay=WEIGHT_DECAY)
    # Re-initialize scheduler with the new optimizer if weight_decay was added after pipeline init
    sparse_rag_pipeline.scheduler = get_cosine_schedule_with_warmup(
            sparse_rag_pipeline.optimizer,
            num_warmup_steps=int(0.1 * total_scheduler_steps),
            num_training_steps=total_scheduler_steps
        )


    # WandB watch (optional, can add overhead, log less frequently if needed)
    if wandb.run:
        wandb.watch(sparse_rag_pipeline.generator, log="all", log_freq=max(100, num_batches_per_epoch_calc // 2))


    # --- Training Loop ---
    logger.info(f"Starting Sparse RAG Generator Training for {EPOCHS} epochs...")
    best_eval_f1_score = 0.0
    cumulative_training_losses = [] # To store all batch losses across all epochs

    for epoch_count in range(EPOCHS):
        current_epoch_for_logging = epoch_count + 1
        logger.info(f"\n--- Training Epoch {current_epoch_for_logging}/{EPOCHS} ---")
        
        # train_pipeline now takes the pre-retrieved dataset
        batch_losses_this_epoch = sparse_rag_pipeline.train_pipeline(
            dataset_pre_retrieved=train_data, # Pass the pre-retrieved data
            batch_size=BATCH_SIZE,
            epochs=1, # train_pipeline is called per epoch, so it runs for 1 internal epoch
            stage_name=f"GenTrain_Epoch{current_epoch_for_logging}"
        )
        if batch_losses_this_epoch: # If list is not empty
            cumulative_training_losses.extend(batch_losses_this_epoch)
            logger.info(f"Epoch {current_epoch_for_logging} average training batch loss: {np.mean(batch_losses_this_epoch):.4f}")
        else:
            logger.warning(f"Epoch {current_epoch_for_logging} reported no batch losses.")


        # --- Evaluation ---
        if dev_data:
            logger.info(f"--- Evaluating after Epoch {current_epoch_for_logging} ---")
            # Evaluation uses pipeline.generate_answer(), which uses the live XapianRetriever
            # The dev_data here is used to provide queries and reference answers.
            # If dev_data also contains "retrieved_contexts", evaluate_pipeline should ignore them
            # and let the pipeline do its own live retrieval for evaluation.
            # For this, evaluate_pipeline should ideally just take queries/answers from dev_data.
            # Let's assume evaluate_pipeline can handle dicts with 'query' and 'answers' keys.
            eval_scores = evaluate_pipeline_sparse(
                pipeline=sparse_rag_pipeline,
                test_set=dev_data, # This will contain pre-retrieved contexts if EVAL_WITH_PRE_RETRIEVED_CONTEXTS is True
                verbose=False,
                log_path=os.path.join(CHECKPOINTS_DIR, f"predictions_epoch{current_epoch_for_logging}.json"),
                top_k=K_RETRIEVAL_INFERENCE,
                epoch=current_epoch_for_logging,
                use_pre_retrieved_contexts_for_eval=EVAL_WITH_PRE_RETRIEVED_CONTEXTS, # Pass the flag
                pre_retrieved_contexts_key="retrieved_contexts" # Matches key from offline script
            )
            logger.info(f"Evaluation (Epoch {current_epoch_for_logging}) — EM: {eval_scores['EM']:.2f}%, F1: {eval_scores['F1']:.2f}%")
            if wandb.run:
                wandb.log({
                    "Eval_EM_Epoch": eval_scores["EM"],
                    "Eval_F1_Epoch": eval_scores["F1"],
                    "epoch_completed_eval": current_epoch_for_logging
                })

            if eval_scores["F1"] > best_eval_f1_score:
                best_eval_f1_score = eval_scores["F1"]
                logger.info(f"🏆 New best F1 score on Dev: {best_eval_f1_score:.2f}%. Saving model to {BEST_MODEL_DIR}...")
                sparse_rag_pipeline.save_model(BEST_MODEL_DIR)
        else:
            logger.warning("No development data loaded, skipping evaluation.")

    logger.info("--- Training complete. ---")

    # --- Save Final Model ---
    logger.info(f"Saving final model components to '{FINAL_MODEL_DIR}'...")
    sparse_rag_pipeline.save_model(FINAL_MODEL_DIR)
    if wandb.run: # Log path to final model if using wandb
        wandb.config.update({"final_model_path": FINAL_MODEL_DIR})
    logger.info("Final model saved.")

    # --- Plotting Training Loss (Optional) ---
    if cumulative_training_losses:
        plt.figure(figsize=(12, 7))
        plt.plot(cumulative_training_losses, label='Training Loss (Per Batch)')
        plt.xlabel("Training Batch (Cumulative)")
        plt.ylabel("Loss")
        plt.title("Sparse RAG (Option A, TrueBatch) Generator Training Loss")
        plt.legend()
        plt.grid(True)
        loss_plot_filename = "training_loss_sparse_rag_optA_TB.png"
        loss_plot_path = os.path.join(CHECKPOINTS_DIR, loss_plot_filename)
        try:
            plt.savefig(loss_plot_path)
            logger.info(f"Training loss plot saved to {loss_plot_path}")
            if wandb.run:
                wandb.log({"training_loss_plot": wandb.Image(loss_plot_path)})
        except Exception as e:
            logger.error(f"Failed to save training loss plot: {e}", exc_info=True)


    if wandb.run: # Finish wandb run
        wandb.finish()
    logger.info("\n--- Main application script execution finished. ---")

if __name__ == "__main__":
    main_app_sparse_option_a()