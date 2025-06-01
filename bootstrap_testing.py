import os

from xapian_eval import evaluate_sparse_rag_pipeline
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

import pandas as pd
import wandb

from NqDataset import NQDataset
from QuestionEncoder import QuestionEncoder
from DenseRetriever import DenseRetriever
from utils import load_local_nq_json, custom_collate_fn, load_precomputed_sparse_results
from RagEval import evaluate_custom_rag_model, evaluate_dense_rag_model


N_BOOTSTRAP_ITERATIONS = 10
EVAL_DATA_FILE = "downloads/data/gold_passages_info/nq_dev.json"
SPARSE_RESULTS_FILE = "downloads/data/nq_dev_sparse_retrieval.jsonl"

retriever_e5_model_name = "models/retriever_finetuned_e5_best"
question_encoder_path = "rag_train_hybrid_v4/best_model/question_encoder"
question_tokenizer_path = "rag_train_hybrid_v4/best_model/question_tokenizer"
generator_tokenizer_path = "rag_train_hybrid_v4/best_model/generator_tokenizer"
generator_path = "rag_train_hybrid_v4/best_model/bart_generator"

FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"Using device: {device}")


wandb_run = wandb.init(
    project="RAG_Bootstrap_Testing", # Or your preferred project name
    name=f"bootstrap_eval_{N_BOOTSTRAP_ITERATIONS}_iters",
    config={
        "n_bootstrap_iterations": N_BOOTSTRAP_ITERATIONS,
        "eval_data_file": EVAL_DATA_FILE,
        "sparse_results_file": SPARSE_RESULTS_FILE,
        "question_encoder_path": question_encoder_path,
        "generator_path": generator_path,
        "retriever_e5_model_name": retriever_e5_model_name
        }
)

# --- 1. Load Original Evaluation Data ---
print("Loading original evaluation data...")
original_eval_items = load_local_nq_json(EVAL_DATA_FILE, limit=None)
# Load precomputed sparse results
sparse_results_lookup = load_precomputed_sparse_results(SPARSE_RESULTS_FILE)
print(f"Loaded {len(original_eval_items)} evaluation items.")

# --- 2. Load All Models, Tokenizers, and Retrievers ---
print("Loading models and retrievers...")
# 1. Initialize Tokenizers
print(f"Loading E5 question tokenizer from: {question_tokenizer_path}")
e5_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_path)
print(f"Loading BART generator tokenizer from: {generator_tokenizer_path}")
bart_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_path)
print("Tokenizers initialized.")

# 2. Initialize Models
print(f"Loading E5 Question Encoder: {question_encoder_path}")
question_encoder = QuestionEncoder.from_pretrained(question_encoder_path).to(device)
question_encoder.eval() 
print("E5 Question Encoder loaded.")

print(f"Loading BART Generator: {generator_path}")
generator = AutoModelForSeq2SeqLM.from_pretrained(generator_path).to(device)
if hasattr(generator.config, "forced_bos_token_id") and \
    generator.config.forced_bos_token_id is None and \
    generator.config.bos_token_id is not None:
    generator.config.forced_bos_token_id = generator.config.bos_token_id
    print(f"Set generator.config.forced_bos_token_id to {generator.config.bos_token_id}")
print("BART Generator loaded.")

# 3. Initialize DenseRetriever
print(f"Initializing custom DenseRetriever with E5: {retriever_e5_model_name}")
dense_retriever_instance = DenseRetriever(
    FAISS_INDEX_PATH, METADATA_PATH, device, retriever_e5_model_name,
    ef_search=1500, ef_construction=200, fine_tune=False, doc_encoder_model=retriever_e5_model_name)
print("DenseRetriever initialized.")

loaded_components = {
    "generator": generator, "bart_tokenizer": bart_tokenizer,
    "question_encoder": question_encoder, "e5_tokenizer": e5_tokenizer,
    "dense_retriever": dense_retriever_instance,
    "sparse_results_lookup": sparse_results_lookup, "device": device
}

eval_config = {
    "k_dense_for_hybrid" : 100,
    "k_retrieved_for_generator" : 50,
    "MAX_QUESTION_LENGTH" : 128,
    "MAX_ANSWER_LENGTH" : 64,
    "EVAL_BATCH_SIZE" : 4,
    "MAX_COMBINED_LENGTH_FOR_GEN" : 512,
    "EVAL_BATCH_SIZE" : 4
}

wandb_run.config.update({"eval_config": eval_config})

def evaluate_bootstrap_sample(bootstrap_data_items, pipeline_type, components, eval_config):
    """
    Evaluate a bootstrap sample of the evaluation data.
    """
    dataset = NQDataset(
        data_list=bootstrap_data_items,
        sparse_retrieval_data=components["sparse_results_lookup"],
        question_tokenizer=components["e5_tokenizer"], # Assuming e5_tokenizer for QuestionEncoder
        generator_tokenizer=components["bart_tokenizer"],
        max_question_length=eval_config["MAX_QUESTION_LENGTH"],
        max_answer_length=eval_config["MAX_ANSWER_LENGTH"]
    )
    dataloader = DataLoader(dataset, batch_size=eval_config["EVAL_BATCH_SIZE"], collate_fn=custom_collate_fn)

    metrics = {}
    if pipeline_type == "hybrid":
        metrics, _ = evaluate_custom_rag_model(
            question_encoder_model=components["question_encoder"],
            dense_retriever=components["dense_retriever"],
            generator_model=components["generator"],
            eval_dataloader=dataloader,
            question_tokenizer=components["e5_tokenizer"],
            generator_tokenizer=components["bart_tokenizer"],
            k_retrieved=eval_config["k_retrieved_for_generator"],
            max_combined_length=eval_config["MAX_COMBINED_LENGTH_FOR_GEN"],
            max_answer_length=eval_config["MAX_ANSWER_LENGTH"],
            device=components["device"],
            K_DENSE_RETRIEVAL=eval_config["k_dense_for_hybrid"],
            wandb_run_obj=None
        )
    elif pipeline_type == "dense":
        metrics, _ = evaluate_dense_rag_model(
            question_encoder_model=components["question_encoder"],
            dense_retriever=components["dense_retriever"],
            generator_model=components["generator"],
            eval_dataloader=dataloader,
            question_tokenizer=components["e5_tokenizer"],
            generator_tokenizer=components["bart_tokenizer"],
            k_retrieved=eval_config["k_retrieved_for_generator"],
            max_combined_length=eval_config["MAX_COMBINED_LENGTH_FOR_GEN"],
            max_answer_length=eval_config["MAX_ANSWER_LENGTH"],
            device=components["device"],
            K_DENSE_RETRIEVAL=eval_config["k_dense_for_hybrid"],
            wandb_run_obj=None
        )
    elif pipeline_type == "xapian":
        metrics, _ = evaluate_sparse_rag_pipeline(
            generator_model=components["generator"],
            eval_dataloader=dataloader,
            generator_tokenizer=components["bart_tokenizer"],
            k_to_feed_generator=eval_config["k_retrieved_for_generator"],
            max_combined_length=eval_config["MAX_COMBINED_LENGTH_FOR_GEN"],
            max_answer_length=eval_config["MAX_ANSWER_LENGTH"],
            device=components["device"],
            wandb_run_obj=None
        )

    return metrics.get("f1", 0.0), metrics.get("exact_match", 0.0)
        
N = len(original_eval_items)
if N == 0:
    print("No evaluation data loaded. Exiting.")
    exit()

all_results = {
    "hybrid": {"f1": [], "em": []},
    "dense": {"f1": [], "em": []},
    "xapian": {"f1": [], "em": []}
}

iteration_data_log = []

for pipeline_name in all_results.keys():
    print(f"\nBootstrapping for pipeline: {pipeline_name}")
    loaded_components["question_encoder"].eval()
    loaded_components["generator"].eval()

    if "dense_retriever" in loaded_components:
        loaded_components["dense_retriever"].query_encoder.eval()
    

    for i in range(N_BOOTSTRAP_ITERATIONS):
        # Create bootstrap sample by resampling original_eval_items
        bootstrap_sample_items = random.choices(original_eval_items, k=N)

        f1, em = evaluate_bootstrap_sample(bootstrap_sample_items, pipeline_name, loaded_components, eval_config)
        all_results[pipeline_name]["f1"].append(f1)
        all_results[pipeline_name]["em"].append(em)

        # Log per-iteration results to wandb
        keys_list = list(all_results.keys())
        wandb_run.log({
            f"bootstrap/{pipeline_name}/f1": f1,
            f"bootstrap/{pipeline_name}/em": em,
            # "bootstrap_iteration": i + 1 
        }, step=i + 1 + (keys_list.index(pipeline_name) * N_BOOTSTRAP_ITERATIONS) ) 

        iteration_data_log.append({
            "pipeline": pipeline_name,
            "iteration": i + 1,
            "f1": f1,
            "em": em
        })

        if (i + 1) % (N_BOOTSTRAP_ITERATIONS // 10) == 0: # Log progress
            print(f"  Iteration {i+1}/{N_BOOTSTRAP_ITERATIONS} for {pipeline_name} done. Last F1: {f1:.4f}, EM: {em:.4f}")
                
# --- 5. Analyze and Log Final Results ---
print("\n--- Bootstrap Testing Results ---")
summary_metrics_for_wandb = {}

iteration_df = pd.DataFrame(iteration_data_log)
wandb_run.log({"bootstrap_iteration_details": wandb.Table(dataframe=iteration_df)})


for pipeline_name, scores_dict in all_results.items():
    if not scores_dict["f1"]:
        print(f"\nNo scores collected for {pipeline_name} pipeline. Skipping summary.")
        continue
        
    f1_scores = np.array(scores_dict["f1"])
    em_scores = np.array(scores_dict["em"])

    mean_f1 = np.mean(f1_scores)
    median_f1 = np.median(f1_scores)
    std_err_f1 = np.std(f1_scores) 
    f1_conf_interval = np.percentile(f1_scores, [2.5, 97.5])

    mean_em = np.mean(em_scores)
    median_em = np.median(em_scores)
    std_err_em = np.std(em_scores)
    em_conf_interval = np.percentile(em_scores, [2.5, 97.5])

    print(f"\nPipeline: {pipeline_name.upper()}")
    print(f"  F1 Scores ({len(f1_scores)} samples):")
    print(f"    Mean: {mean_f1:.4f}")
    print(f"    Median: {median_f1:.4f}")
    print(f"    Standard Error: {std_err_f1:.4f}")
    print(f"    95% Confidence Interval: [{f1_conf_interval[0]:.4f}, {f1_conf_interval[1]:.4f}]")

    print(f"  Exact Match Scores ({len(em_scores)} samples):")
    print(f"    Mean: {mean_em:.4f}")
    print(f"    Median: {median_em:.4f}")
    print(f"    Standard Error: {std_err_em:.4f}")
    print(f"    95% Confidence Interval: {np.percentile(em_scores, [2.5, 97.5])}")

    # Log summary to wandb
    summary_metrics_for_wandb[f"summary/{pipeline_name}/f1_mean"] = mean_f1
    summary_metrics_for_wandb[f"summary/{pipeline_name}/f1_median"] = median_f1
    summary_metrics_for_wandb[f"summary/{pipeline_name}/f1_std_err"] = std_err_f1
    summary_metrics_for_wandb[f"summary/{pipeline_name}/f1_ci_low"] = f1_conf_interval[0]
    summary_metrics_for_wandb[f"summary/{pipeline_name}/f1_ci_high"] = f1_conf_interval[1]
    summary_metrics_for_wandb[f"summary/{pipeline_name}/em_mean"] = mean_em
    summary_metrics_for_wandb[f"summary/{pipeline_name}/em_median"] = median_em
    summary_metrics_for_wandb[f"summary/{pipeline_name}/em_std_err"] = std_err_em
    summary_metrics_for_wandb[f"summary/{pipeline_name}/em_ci_low"] = em_conf_interval[0]
    summary_metrics_for_wandb[f"summary/{pipeline_name}/em_ci_high"] = em_conf_interval[1]

    # Log histograms of the scores
    wandb_run.log({
        f"histograms/{pipeline_name}/f1_distribution": wandb.Histogram(f1_scores),
        f"histograms/{pipeline_name}/em_distribution": wandb.Histogram(em_scores)
    })

