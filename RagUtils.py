import torch
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import torch.nn.functional as F
from torch import nn

def retrieve_documents_for_batch(
    query_embeddings_batch: torch.Tensor, # Shape: [batch_size, query_embed_dim]
    dense_retriever,      
    k: int,                              
    normalize_query_for_faiss: bool = True 
) -> Dict[str, Any]:
    """
    Retrieves documents and their embeddings for a batch of query embeddings.

    Args:
        query_embeddings_batch: Batch of query embeddings from the question encoder.
        dense_retriever: Initialized instance of your DenseRetriever.
        k: Number of top documents to retrieve for each query.
        normalize_query_for_faiss: Whether to L2 normalize query embeddings before FAISS search.
                                   E5 often uses normalized embeddings with inner product.

    Returns:
        A dictionary containing:
            "retrieved_doc_texts": List[List[str]] (batch_size, k)
            "retrieved_doc_titles": List[List[str]] (batch_size, k)
            "retrieved_doc_embeddings": torch.Tensor (batch_size, k, doc_embed_dim)
            "retrieved_doc_faiss_ids": np.ndarray (batch_size, k)
            "retrieved_doc_faiss_distances": np.ndarray (batch_size, k)
    """
    if not isinstance(query_embeddings_batch, torch.Tensor):
        raise TypeError("query_embeddings_batch must be a PyTorch Tensor.")

    batch_size = query_embeddings_batch.shape[0]
    
    # 1. Prepare query embeddings for FAISS search
    query_embeddings_np = query_embeddings_batch.detach().cpu().numpy()

    if normalize_query_for_faiss and "e5" in dense_retriever.model_name.lower():
        # FAISS normalize_L2 normalizes in-place
        faiss.normalize_L2(query_embeddings_np)

    # 2. Perform batch FAISS search
    # `dense_retriever.index` is your loaded FAISS index
    # It returns distances and indices (FAISS IDs)
    distances_batch, faiss_ids_batch = dense_retriever.index.search(query_embeddings_np, k)
    # distances_batch shape: [batch_size, k]
    # faiss_ids_batch shape: [batch_size, k]

    # Initialize lists to store outputs for the batch
    batch_doc_texts: List[List[str]] = [[] for _ in range(batch_size)]
    batch_doc_titles: List[List[str]] = [[] for _ in range(batch_size)]
    
    # Collect all texts that need to be embedded by the doc_encoder
    # This helps in batching the encoding process
    all_texts_to_embed_flat: List[str] = []
    # Keep track of (batch_index, doc_index_in_batch) for re-assigning embeddings
    embedding_map_indices: List[Tuple[int, int]] = [] 
    # Store actual valid doc texts that were found, to avoid encoding placeholders unnecessarily
    valid_doc_texts_for_encoding: List[str] = []


    for i in range(batch_size): # For each query in the batch
        for j in range(k):      # For each retrieved doc for that query
            faiss_id = faiss_ids_batch[i, j]
            
            if faiss_id == -1: # FAISS can return -1 if fewer than k results are found
                batch_doc_texts[i].append("")  # Placeholder for missing doc
                batch_doc_titles[i].append("") # Placeholder for missing title
                # We will add a placeholder (e.g., zero vector) for its embedding later
                all_texts_to_embed_flat.append("") # Add empty string to maintain structure for embedding map
                embedding_map_indices.append((i,j))
            else:
                metadata_entry = dense_retriever.metadata.get(str(faiss_id))
                if metadata_entry:
                    text = metadata_entry.get("text", "")
                    title = metadata_entry.get("title", "")
                    batch_doc_texts[i].append(text)
                    batch_doc_titles[i].append(title)
                    all_texts_to_embed_flat.append(text) # Text to be encoded
                    embedding_map_indices.append((i,j))
                    if text: # Only add non-empty texts for actual encoding
                        valid_doc_texts_for_encoding.append(text)
                else:
                    # logger_retriever.warning(f"Metadata not found for FAISS ID: {faiss_id}")
                    batch_doc_texts[i].append("")
                    batch_doc_titles[i].append("")
                    all_texts_to_embed_flat.append("")
                    embedding_map_indices.append((i,j))

    # 3. Batch embed all collected valid document texts
    doc_embeddings_flat_tensor = None
    if valid_doc_texts_for_encoding:
        # Use the doc_encoder from your DenseRetriever instance
        # This doc_encoder is a SentenceTransformer model
        dense_retriever.doc_encoder.eval() # Ensure it's in eval mode (it's frozen anyway)
        with torch.no_grad():
            # Apply prefixes if your doc_encoder (E5) expects them for passages
            prefixed_valid_texts = dense_retriever._apply_model_specific_prefixes(
                valid_doc_texts_for_encoding, 
                text_type="passage"
            )
            doc_embeddings_flat_tensor = dense_retriever.doc_encoder.encode(
                prefixed_valid_texts,
                convert_to_tensor=True,
                # Normalize if your RAG loss expects normalized doc embeddings
                # For E5, if queries are normalized, docs should be too for dot product.
                normalize_embeddings=("e5" in dense_retriever.model_name.lower()), 
                show_progress_bar=False, # Optional
                device=dense_retriever.device # Ensure encoding happens on the correct device
            ) # Shape: [num_valid_docs, doc_embed_dim]

    # 4. Reshape document embeddings and handle placeholders
    doc_embed_dim = dense_retriever.doc_encoder.get_sentence_embedding_dimension()
    batch_doc_embeddings = torch.zeros(
        (batch_size, k, doc_embed_dim), 
        dtype=torch.float, 
        device=dense_retriever.device
    )

    valid_doc_idx_counter = 0
    for original_flat_idx, (batch_i, doc_j) in enumerate(embedding_map_indices):
        # If the text at all_texts_to_embed_flat[original_flat_idx] was valid and encoded
        if all_texts_to_embed_flat[original_flat_idx] and doc_embeddings_flat_tensor is not None and valid_doc_idx_counter < doc_embeddings_flat_tensor.shape[0]:
            batch_doc_embeddings[batch_i, doc_j, :] = doc_embeddings_flat_tensor[valid_doc_idx_counter]
            valid_doc_idx_counter += 1
        # Else, it remains a zero vector (placeholder for missing/empty doc)

    return {
        "retrieved_doc_texts": batch_doc_texts,
        "retrieved_doc_titles": batch_doc_titles,
        "retrieved_doc_embeddings": batch_doc_embeddings, 
        "retrieved_doc_faiss_ids": faiss_ids_batch,       
        "retrieved_doc_faiss_distances": distances_batch 
    }


def prepare_generator_inputs(
    original_question_strings: List[str],       # List of raw question strings for the batch [batch_size]
    retrieved_doc_titles: List[List[str]],      # From retrieve_documents_for_batch output [batch_size, k_retrieved]
    retrieved_doc_texts: List[List[str]],       # From retrieve_documents_for_batch output [batch_size, k_retrieved]
    generator_tokenizer: PreTrainedTokenizerFast, # Initialized BART tokenizer
    max_combined_length: int,                   # Max sequence length for BART encoder (e.g., 512 or 1024)
    device: torch.device                        # Device to put output tensors on
) -> Dict[str, torch.Tensor]:
    """
    Prepares formatted and tokenized inputs for the generator model's encoder.
    Each question is combined with each of its k retrieved documents.

    Args:
        original_question_strings: List of raw question strings.
        retrieved_doc_titles: List of lists of retrieved document titles.
        retrieved_doc_texts: List of lists of retrieved document texts.
        generator_tokenizer: The tokenizer for the generator model (e.g., BART).
        max_combined_length: The maximum sequence length for the generator's input.
        device: The torch device for the output tensors.

    Returns:
        A dictionary containing:
            "generator_input_ids": torch.Tensor [batch_size * k_retrieved, max_combined_length]
            "generator_attention_mask": torch.Tensor [batch_size * k_retrieved, max_combined_length]
    """
    batch_size = len(original_question_strings)
    if not batch_size or not retrieved_doc_texts or not retrieved_doc_texts[0]:
        # Handle empty input gracefully
        return {
            "generator_input_ids": torch.empty((0, max_combined_length), dtype=torch.long, device=device),
            "generator_attention_mask": torch.empty((0, max_combined_length), dtype=torch.long, device=device)
        }
        
    k_retrieved = len(retrieved_doc_texts[0])
    all_formatted_strings_for_generator = []

    for i in range(batch_size):
        question_str = original_question_strings[i]
        if i >= len(retrieved_doc_titles) or i >= len(retrieved_doc_texts):
            # This case should ideally not happen if data is consistent
            print(f"Warning: Mismatch between questions and retrieved docs for batch item {i}. Skipping.")
            for _ in range(k_retrieved): # Add placeholders to maintain structure if absolutely needed
                 all_formatted_strings_for_generator.append(question_str + " [SEP_TITLE] [SEP_TEXT] ") # Minimal placeholder
            continue

        for j in range(k_retrieved):
            # Ensure titles and texts lists are long enough for doc j
            doc_title = retrieved_doc_titles[i][j] if j < len(retrieved_doc_titles[i]) else ""
            doc_text = retrieved_doc_texts[i][j] if j < len(retrieved_doc_texts[i]) else ""
            
            # RAG-style formatting: question <sep> title <sep> passage
            # Using simple textual separators. The generator_tokenizer will handle BOS/EOS.
            # You might want to experiment with using generator_tokenizer.sep_token if needed,
            # but often a clear textual separation is fine.
            formatted_string = f"{question_str} [SEP_TITLE] {doc_title} [SEP_TEXT] {doc_text}"
            all_formatted_strings_for_generator.append(formatted_string)

    # Tokenize all (batch_size * k_retrieved) formatted strings
    tokenized_generator_inputs = generator_tokenizer(
        all_formatted_strings_for_generator,
        max_length=max_combined_length,
        padding="max_length", # Pad to max_combined_length
        truncation=True,
        return_tensors="pt"
    )

    return {
        "generator_input_ids": tokenized_generator_inputs.input_ids.to(device),
        "generator_attention_mask": tokenized_generator_inputs.attention_mask.to(device)
    }

def calculate_rag_loss(
    query_embeddings: torch.Tensor,              # [batch_size, query_embed_dim]
    retrieved_doc_embeddings: torch.Tensor,    # [batch_size, n_docs, doc_embed_dim]
    generator_input_ids: torch.Tensor,         # [batch_size * n_docs, gen_encoder_seq_len]
    generator_attention_mask: torch.Tensor,    # [batch_size * n_docs, gen_encoder_seq_len]
    target_labels: torch.Tensor,               # [batch_size, target_ans_seq_len]
    generator_model,          # Your BART generator model
    generator_pad_token_id: int,
    n_docs: int,                                # Number of retrieved documents per query
    device: torch.device
) -> torch.Tensor:
    """
    Calculates the RAG marginal negative log-likelihood loss.
    Loss = -log( sum_{i=1 to n_docs} [ P(doc_i | question) * P(answer | question, doc_i) ] )
    """
    batch_size = query_embeddings.shape[0]
    target_answer_seq_len = target_labels.shape[1]
    generator_vocab_size = generator_model.config.vocab_size

    # Ensure all tensors are on the same device as the generator_model
    target_device = device
    query_embeddings = query_embeddings.to(target_device)
    retrieved_doc_embeddings = retrieved_doc_embeddings.to(target_device)
    generator_input_ids = generator_input_ids.to(target_device)
    generator_attention_mask = generator_attention_mask.to(target_device)
    target_labels = target_labels.to(target_device)

    # 1. Calculate Document Probabilities/Scores: P(doc_i | question)
    #    doc_scores shape: [batch_size, n_docs]
    
    # Expand query_embeddings for bmm: [batch_size, 1, query_embed_dim]
    expanded_query_embeddings = query_embeddings.unsqueeze(1)
    
    # Dot product: (B, 1, D_q) @ (B, D_doc, N_docs) -> (B, 1, N_docs) if D_q == D_doc
    # Transpose retrieved_doc_embeddings: [batch_size, doc_embed_dim, n_docs]
    doc_scores = torch.bmm(expanded_query_embeddings, retrieved_doc_embeddings.transpose(1, 2)).squeeze(1)
    # doc_scores shape: [batch_size, n_docs]
    
    # Get log probabilities for documents (log P(doc_i | question))
    doc_log_probs = F.log_softmax(doc_scores, dim=-1) # Shape: [batch_size, n_docs]

    # 2. Prepare inputs for the generator's decoder & get generator logits
    #    The generator_input_ids and generator_attention_mask are for the encoder part.
    #    The target_labels need to be expanded for the n_docs dimension and then
    #    used to create decoder_input_ids and processed for loss calculation.
    
    # Expand target_labels for each document: [batch_size * n_docs, target_ans_seq_len]
    expanded_target_labels = target_labels.repeat_interleave(n_docs, dim=0)
    
    # Set padding tokens in labels to -100 for CrossEntropyLoss
    processed_expanded_labels = expanded_target_labels.clone()
    processed_expanded_labels[processed_expanded_labels == generator_pad_token_id] = -100
    
    # Get logits from the generator
    # The BART model handles creating decoder_input_ids from labels internally
    # when labels are provided to its forward method.
    generator_outputs = generator_model(
        input_ids=generator_input_ids,
        attention_mask=generator_attention_mask,
        labels=processed_expanded_labels # Pass processed labels directly
    )
    # sequence_logits shape: [batch_size * n_docs, target_ans_seq_len, vocab_size]
    sequence_logits = generator_outputs.logits

    # 3. Calculate NLL of the answer given each (question, doc_i) context
    #    NLL(answer | question, doc_i)
    #    Use CrossEntropyLoss with reduction='none' to get per-token losses
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
    # Reshape logits: [ (batch_size * n_docs * target_ans_seq_len), vocab_size ]
    # Reshape labels: [ (batch_size * n_docs * target_ans_seq_len) ]
    per_token_nll = loss_fct(
        sequence_logits.view(-1, generator_vocab_size),
        processed_expanded_labels.view(-1)
    )
    # Reshape per_token_nll back: [batch_size * n_docs, target_ans_seq_len]
    per_token_nll = per_token_nll.view(batch_size * n_docs, target_answer_seq_len)
    
    # Sum per-token NLLs to get per-sequence NLL for each doc context
    # We need to sum only over non-ignored tokens. The loss_fct already gives 0 for -100.
    # Create a mask for non-ignored tokens in labels to correctly sum lengths if needed,
    # but sum() directly should work as -100 tokens have 0 loss.
    nll_per_doc_sequence = per_token_nll.sum(dim=1) # Shape: [batch_size * n_docs]
    
    # Reshape to [batch_size, n_docs]
    nll_per_doc_sequence = nll_per_doc_sequence.view(batch_size, n_docs)

    # 4. Marginalize to get the final RAG loss
    # log P(answer | question, doc_i) = - NLL(answer | question, doc_i)
    log_prob_answer_given_doc = -nll_per_doc_sequence # Shape: [batch_size, n_docs]
    
    # Combine with document log probabilities: log P(doc_i|q) + log P(ans|q,doc_i)
    total_log_probs_for_sumexp = doc_log_probs + log_prob_answer_given_doc # Shape: [batch_size, n_docs]
    
    # logsumexp over the n_docs dimension: log( sum_i exp(total_log_probs_for_sumexp_i) )
    marginal_log_likelihood_per_question = torch.logsumexp(total_log_probs_for_sumexp, dim=1) # Shape: [batch_size]
    
    # Final loss is the negative of the mean marginal log likelihood
    final_loss = -marginal_log_likelihood_per_question.mean() # Scalar

    return final_loss

def hybrid_retrieve_documents_for_batch(
    query_embeddings_batch: torch.Tensor,
    batch_precomputed_sparse_docs: List[List[Dict[str, Any]]], # From DataLoader [batch_size, k_sparse_precomputed]
    dense_retriever,
    final_k: int,
    k_dense_to_fetch: int,
    device: torch.device,
    fusion_k_constant: int = 60 # For RRF
) -> Dict[str, Any]:
    batch_size = query_embeddings_batch.shape[0]

    # 1. Perform Online Dense Retrieval (using your existing retrieve_documents_for_batch logic internally)
    dense_retrieval_results = retrieve_documents_for_batch( # Your existing function
        query_embeddings_batch, dense_retriever, k_dense_to_fetch, True
    )
    # dense_retrieval_results is a dict like:
    # { "retrieved_doc_texts": List[List[str]], "retrieved_doc_titles": List[List[str]],
    #   "retrieved_doc_embeddings": torch.Tensor, "retrieved_doc_faiss_ids": np.ndarray,
    #   "retrieved_doc_faiss_distances": np.ndarray }

    final_batch_doc_texts = [[""]*final_k for _ in range(batch_size)]
    final_batch_doc_titles = [[""]*final_k for _ in range(batch_size)]
    final_batch_doc_embeddings = torch.zeros(
        (batch_size, final_k, dense_retriever.doc_encoder.get_sentence_embedding_dimension()),
        dtype=torch.float, device=device
    )

    for i in range(batch_size):
        # --- Fusion Logic (e.g., Reciprocal Rank Fusion - RRF) ---
        # a. Get dense results for current query
        dense_ids = dense_retrieval_results["retrieved_doc_faiss_ids"][i]
        # Convert dense distances to scores (higher is better, e.g., 1/(1+distance) or use dot product if available)

        # b. Get sparse results for current query (these are precomputed)
        sparse_docs_for_query_i = batch_precomputed_sparse_docs[i] # List of {"id", "sparse_score", ...}

        # c. Perform RRF
        rrf_scores = {}
        # Process dense results
        for rank, doc_id in enumerate(dense_ids):
            if doc_id == -1: continue
            doc_id_str = str(doc_id)
            rrf_scores[doc_id_str] = rrf_scores.get(doc_id_str, 0) + 1.0 / (fusion_k_constant + rank + 1)

        # Process sparse results
        for rank, sparse_doc_info in enumerate(sparse_docs_for_query_i):
            doc_id_str = str(sparse_doc_info["id"])
            rrf_scores[doc_id_str] = rrf_scores.get(doc_id_str, 0) + 1.0 / (fusion_k_constant + rank + 1)

        # d. Sort by RRF score and select top final_k
        sorted_fused_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_final_k_ids_str = sorted_fused_doc_ids[:final_k]

        # e. Gather texts, titles, and compute/fetch DENSE embeddings for these top_final_k_ids_str
        current_final_texts = []
        current_final_titles = []
        doc_ids_for_embedding_lookup = [] # IDs for which we need dense embeddings

        # Map ID to its original dense embedding if already retrieved by dense stage
        dense_embeddings_map = {
            str(dense_retrieval_results["retrieved_doc_faiss_ids"][i][j]): dense_retrieval_results["retrieved_doc_embeddings"][i][j]
            for j in range(len(dense_retrieval_results["retrieved_doc_faiss_ids"][i]))
            if dense_retrieval_results["retrieved_doc_faiss_ids"][i][j] != -1
        }

        texts_to_embed_on_the_fly = []
        indices_for_on_the_fly_embeddings = [] # To place them back correctly

        for j, doc_id_str in enumerate(top_final_k_ids_str):
            metadata_entry = dense_retriever.metadata.get(doc_id_str, {})
            title = metadata_entry.get("title", "")
            text = metadata_entry.get("text", "")
            final_batch_doc_texts[i][j] = text
            final_batch_doc_titles[i][j] = title

            if doc_id_str in dense_embeddings_map:
                final_batch_doc_embeddings[i, j, :] = dense_embeddings_map[doc_id_str]
            elif text: # Needs on-the-fly embedding
                texts_to_embed_on_the_fly.append(text)
                indices_for_on_the_fly_embeddings.append((i, j))
            # else it remains zero vector if no text and not in dense map

        if texts_to_embed_on_the_fly:
            dense_retriever.doc_encoder.eval()
            with torch.no_grad():
                prefixed_texts = dense_retriever._apply_model_specific_prefixes(texts_to_embed_on_the_fly, "passage")
                on_the_fly_embeddings = dense_retriever.doc_encoder.encode(
                    prefixed_texts, convert_to_tensor=True,
                    normalize_embeddings=("e5" in dense_retriever.model_name.lower()),
                    device=dense_retriever.device
                )
                for k_emb, (batch_loc_i, batch_loc_j) in enumerate(indices_for_on_the_fly_embeddings):
                    final_batch_doc_embeddings[batch_loc_i, batch_loc_j, :] = on_the_fly_embeddings[k_emb]

    return {
        "retrieved_doc_texts": final_batch_doc_texts,
        "retrieved_doc_titles": final_batch_doc_titles,
        "retrieved_doc_embeddings": final_batch_doc_embeddings, # Dense embeddings for all final_k docs
        # You might want to return the fused IDs and their RRF scores too for logging/debugging
    }