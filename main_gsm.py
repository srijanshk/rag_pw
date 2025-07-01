import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
import wandb
import numpy as np
import pandas as pd
import traceback
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, get_scheduler
from torch.utils.data import DataLoader

# --- Step 1: Import all custom components ---
from DenseRetriever import DenseRetriever 
from QuestionEncoder import QuestionEncoder
from FlagEmbedding import BGEM3FlagModel
from MathDataset import MathDataset, collate_fn, load_gsm8k_from_file
from RagUtils import calculate_rag_loss, retrieve_documents_for_batch, prepare_generator_inputs, calculate_rag_token_loss
from RagEval import evaluate_rag_sequence, evaluate_rag_token
from utils import extract_final_numeric_answer, extract_gsm8k_gold_answer, compute_accuracy

def main():

  config = {
        "retriever_encoder_name": "BAAI/bge-m3",
        "generator_name": "facebook/bart-large-cnn",
        "faiss_index_path": "/local00/student/shakya/openmath_bge-m3_hnsw_index",
        "metadata_path": "/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
        "train_dataset_path": "./thesis_datasets/gsm8k/train.jsonl",
        "eval_dataset_path": "./thesis_datasets/gsm8k/test.jsonl",
        "num_epochs": 10,
        "batch_size": 1,
        "learning_rate": 5e-5,
        "initial_k": 50, 
        "final_k": 5, 
        "max_q_len": 128,
        "max_a_len": 512,
        "max_combined_len": 512,
        "gradient_accumulation_steps": 16,
        "output_dir": "./rag_token_finetuned_gsm8k_v2"
    }
  
  wandb.init(project="rag-token-thesis", name="gsm8k-finetuning", config=config)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # --- Load Models, Tokenizers, and Retriever ---
  print("--- Initializing Models and Components ---")
  embedding_model = BGEM3FlagModel(config["retriever_encoder_name"], use_fp16=True)
  q_encoder_config = AutoConfig.from_pretrained(config["retriever_encoder_name"])
  question_encoder = QuestionEncoder(config=q_encoder_config, model_name_or_path=config["retriever_encoder_name"]).to(device)
  generator_model = AutoModelForSeq2SeqLM.from_pretrained(config["generator_name"]).to(device)
  question_tokenizer = AutoTokenizer.from_pretrained(config["retriever_encoder_name"])
  generator_tokenizer = AutoTokenizer.from_pretrained(config["generator_name"])

  dense_retriever = DenseRetriever(
      embedding_model=embedding_model,
      index_path=config["faiss_index_path"],
      metadata_path=config["metadata_path"],
      device=device
  )
  print("--- All components initialized successfully. ---")

  # --- Load Data ---
  print("\n--- Loading Training and Evaluation Data ---")
  train_data_list = load_gsm8k_from_file(config["train_dataset_path"])
  train_dataset = MathDataset(train_data_list, tokenizer=generator_tokenizer, max_q_len=config["max_q_len"], 
                              max_a_len=config["max_a_len"], question_key='question', answer_key='answer')
  train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=True)
  
  eval_data_list = load_gsm8k_from_file(config["eval_dataset_path"])
  eval_dataset = MathDataset(eval_data_list, tokenizer=generator_tokenizer, max_q_len=config["max_q_len"], 
                              max_a_len=config["max_a_len"], question_key='question', answer_key='answer')
  eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], collate_fn=collate_fn)

  # --- Setup Optimizer and Scheduler ---
  optimizer = optim.AdamW(list(question_encoder.parameters()) + list(generator_model.parameters()), lr=config["learning_rate"])
  num_training_steps = config["num_epochs"] * len(train_loader)
  lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

  # --- Training Loop ---
  print("\n--- Starting RAG-Token End-to-End Training with Evaluation ---")
  os.makedirs(config["output_dir"], exist_ok=True)
  best_accuracy = -1.0

  for epoch in range(config["num_epochs"]):
      # --- Training Phase ---
      question_encoder.train()
      generator_model.train()
      total_loss = 0
      progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} Training")
      for step, batch in enumerate(progress_bar):
          # (Training logic is the same as before...)
          tokenized_questions = question_tokenizer(batch['original_question'], padding=True, truncation=True, return_tensors='pt').to(device)
          query_embeddings = question_encoder(input_ids=tokenized_questions['input_ids'], attention_mask=tokenized_questions['attention_mask'])[0]
          retrieved_batch = dense_retriever.search_batch(batch['original_question'], k=config["initial_k"])
          
          reranked_batch = []
          for i, question in enumerate(batch['original_question']):
              pairs = [[question, doc['solution_chunk']] for doc in retrieved_batch[i]]
              scores = embedding_model.compute_score(pairs)['colbert']
              for doc, score in zip(retrieved_batch[i], scores): doc['rerank_score'] = score
              reranked_batch.append(sorted(retrieved_batch[i], key=lambda x: x['rerank_score'], reverse=True)[:config["final_k"]])
          
          b_texts, b_titles, flat_texts = [], [], []
          for docs in reranked_batch:
              texts, titles = [d['solution_chunk'] for d in docs], [d.get('problem', 'N/A') for d in docs]
              b_texts.append(texts); b_titles.append(titles); flat_texts.extend(texts)

          doc_embeds_flat = embedding_model.encode(flat_texts, return_dense=True)['dense_vecs']
          doc_embeds = torch.from_numpy(doc_embeds_flat).view(len(batch['original_question']), config["final_k"], -1).to(device).to(query_embeddings.dtype)

          gen_inputs = prepare_generator_inputs(batch['original_question'], b_titles, b_texts, generator_tokenizer, config["max_combined_len"], device)
          gen_outputs = generator_model(input_ids=gen_inputs['generator_input_ids'], attention_mask=gen_inputs['generator_attention_mask'])

          labels_tensor = torch.stack(batch['labels']).to(device)
          
          loss = calculate_rag_token_loss(query_embeddings, doc_embeds, gen_outputs, labels_tensor, 
                                          generator_tokenizer.pad_token_id, config["final_k"], device)
          
          total_loss += loss.item()
          loss = loss / config["gradient_accumulation_steps"]
          loss.backward()

          if (step + 1) % config["gradient_accumulation_steps"] == 0:
              optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

          progress_bar.set_postfix({'loss': loss.item() * config["gradient_accumulation_steps"]})
          wandb.log({"step_loss": loss.item() * config["gradient_accumulation_steps"]})
      
      avg_epoch_loss = total_loss / len(train_loader)
      wandb.log({"epoch": epoch + 1, "avg_epoch_loss": avg_epoch_loss})
      print(f"Epoch {epoch + 1} Training Finished. Avg Loss: {avg_epoch_loss:.4f}")

      # --- Evaluation Phase ---
      print(f"\n--- Running Evaluation for Epoch {epoch + 1} ---")
      eval_metrics, logged_examples = evaluate_rag_sequence(
          question_encoder, generator_model, dense_retriever, eval_loader,
          generator_tokenizer, device, k_retrieved=5
      )
      current_accuracy = eval_metrics["EM_accuracy"]
      print(f"Epoch {epoch + 1} Evaluation Finished. Exact Match Accuracy: {current_accuracy:.2f}%")
      log_data = {"epoch": epoch + 1, "eval_accuracy": current_accuracy}
      if logged_examples:
          df = pd.DataFrame(logged_examples)
          log_data[f"evaluation_samples_epoch_{epoch+1}"] = wandb.Table(dataframe=df)
      wandb.log(log_data)

      # --- Save Best Model ---
      if current_accuracy > best_accuracy:
          best_accuracy = current_accuracy
          print(f"New best accuracy! Saving model to best_model directory...")
          best_model_dir = os.path.join(config["output_dir"], "best_model")
          os.makedirs(best_model_dir, exist_ok=True)
          question_encoder.save_pretrained(os.path.join(best_model_dir, "question_encoder"))
          generator_model.save_pretrained(os.path.join(best_model_dir, "generator"))
          question_tokenizer.save_pretrained(os.path.join(best_model_dir, "question_tokenizer"))
          generator_tokenizer.save_pretrained(os.path.join(best_model_dir, "generator_tokenizer"))
          with open(os.path.join(best_model_dir, "best_score.json"), "w") as f:
              json.dump({"best_accuracy": best_accuracy, "epoch": epoch + 1}, f)

  wandb.finish()
  print("--- Training and Evaluation Finished ---")

if __name__ == "__main__":  
  main()
