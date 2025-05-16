# File: HFCompactRetriever.py

import torch
import numpy as np
from transformers import RagConfig
from transformers.models.rag.retrieval_rag import RagRetriever, Index as HFRagIndexBase
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import List, Dict, Tuple, Optional, Union
from transformers.modeling_outputs import ModelOutput

# ... (DummyHFIndex class as before) ...
class DummyHFIndex(HFRagIndexBase):
    def __init__(self):
        self._is_initialized = True
        self._len = 0
    def init_index(self): pass
    def add_vectors(self, embeddings: np.ndarray, doc_dicts: List[dict]): pass
    def save(self, path: str): pass
    def load(self, path: str): pass
    def search_docs(self, query_hidden_states: np.ndarray, k: int) -> Tuple[np.ndarray, List[dict]]:
        batch_size = query_hidden_states.shape[0]
        empty_doc_ids = np.full((batch_size, k), -1, dtype=np.int64)
        empty_doc_dicts = [{"title": ["" for _ in range(k)], "text": ["" for _ in range(k)]} for _ in range(batch_size)]
        return empty_doc_ids, empty_doc_dicts
    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        batch_size = doc_ids.shape[0]
        n_docs_per_query = doc_ids.shape[1] if doc_ids.ndim > 1 and doc_ids.shape[1] > 0 else (1 if doc_ids.ndim == 1 and doc_ids.shape[0] > 0 else 0)
        return [{"title": ["" for _ in range(n_docs_per_query)], "text": ["" for _ in range(n_docs_per_query)]} for _ in range(batch_size)]
    def is_initialized(self) -> bool: return self._is_initialized
    def get_len(self) -> int: return self._len


class HFCompatDenseRetriever(RagRetriever):
    def __init__(self,
                 config: RagConfig,
                 question_encoder_tokenizer: PreTrainedTokenizerBase,
                 generator_tokenizer: PreTrainedTokenizerBase,
                 custom_dense_retriever: 'DenseRetriever',
                 ):
        self.dummy_index_instance = DummyHFIndex()
        super().__init__(config, question_encoder_tokenizer, generator_tokenizer, index=self.dummy_index_instance)
        self.custom_retriever = custom_dense_retriever
        # ... (Your existing attribute checks for custom_retriever) ...
        if not hasattr(self.custom_retriever, "doc_encoder") or \
           not hasattr(self.custom_retriever.doc_encoder, "encode") or \
           not hasattr(self.custom_retriever.doc_encoder, "get_sentence_embedding_dimension"):
            raise ValueError("custom_dense_retriever must have 'doc_encoder' with 'encode' and 'get_sentence_embedding_dimension'")
        if not hasattr(self.custom_retriever, "index") or \
           not hasattr(self.custom_retriever.index, "search"):
            raise ValueError("custom_dense_retriever must have FAISS 'index' with 'search' method.")
        if not hasattr(self.custom_retriever, "metadata") or not isinstance(self.custom_retriever.metadata, dict):
            raise ValueError("custom_dense_retriever must have 'metadata' as a dictionary.")
        if not hasattr(self.custom_retriever, "model_name"):
            raise ValueError("custom_dense_retriever must have 'model_name' attribute.")
        if not hasattr(self.custom_retriever, "_apply_model_specific_prefixes"):
             raise ValueError("custom_dense_retriever must have '_apply_model_specific_prefixes' method.")
        if not hasattr(self.custom_retriever, "device"):
             raise ValueError("custom_dense_retriever must have 'device' attribute.")


    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norm == 0, 1e-9, norm)

    def retrieve(self,
                 question_hidden_states_np: np.ndarray,
                 n_docs: int
                ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, List[str]]]]:
        # ... (Keep your existing retrieve logic that expects NumPy arrays and returns the 3 items) ...
        query_embeddings_np = question_hidden_states_np
        if "e5" in self.custom_retriever.model_name.lower():
             query_embeddings_np = self._normalize_embeddings(query_embeddings_np)
        distances_batch, indices_batch = self.custom_retriever.index.search(query_embeddings_np, n_docs)
        batch_size_times_beams = query_embeddings_np.shape[0]
        doc_ids_list_batch_for_np = np.full((batch_size_times_beams, n_docs), -1, dtype=np.int64)
        formatted_docs_for_rag = []
        doc_texts_to_encode_flat = []
        doc_encode_mapping = []
        for i in range(batch_size_times_beams):
            titles_for_current_query = [""] * n_docs
            texts_for_current_query = [""] * n_docs
            for j in range(n_docs):
                doc_faiss_id = indices_batch[i, j]
                current_text, current_title, is_placeholder = "", "", True
                if doc_faiss_id != -1:
                    metadata = self.custom_retriever.metadata.get(str(doc_faiss_id), {})
                    text_from_meta, title_from_meta = metadata.get("text", ""), metadata.get("title", "")
                    if isinstance(text_from_meta, str) and len(text_from_meta.strip()) > 0:
                        doc_ids_list_batch_for_np[i, j] = doc_faiss_id
                        current_text, current_title, is_placeholder = text_from_meta, title_from_meta, False
                titles_for_current_query[j], texts_for_current_query[j] = current_title, current_text
                doc_texts_to_encode_flat.append(current_text)
                doc_encode_mapping.append((i, j, is_placeholder))
            formatted_docs_for_rag.append({"title": titles_for_current_query, "text": texts_for_current_query})
        embedding_dim = self.custom_retriever.doc_encoder.get_sentence_embedding_dimension()
        retrieved_doc_embeds_batch_np = np.zeros((batch_size_times_beams, n_docs, embedding_dim), dtype=np.float32)
        if doc_texts_to_encode_flat:
            self.custom_retriever.doc_encoder.eval()
            with torch.no_grad():
                prefixed_doc_texts = self.custom_retriever._apply_model_specific_prefixes(doc_texts_to_encode_flat, "passage")
                all_doc_embeds_tensor = self.custom_retriever.doc_encoder.encode(
                    prefixed_doc_texts, convert_to_tensor=True,
                    normalize_embeddings="e5" in self.custom_retriever.model_name.lower(),
                    show_progress_bar=False, device=self.custom_retriever.device
                )
            all_doc_embeds_np = all_doc_embeds_tensor.cpu().numpy()
            current_flat_idx = 0
            for batch_idx, doc_idx, _ in doc_encode_mapping:
                if current_flat_idx < len(all_doc_embeds_np):
                    retrieved_doc_embeds_batch_np[batch_idx, doc_idx, :] = all_doc_embeds_np[current_flat_idx]
                current_flat_idx += 1
        return retrieved_doc_embeds_batch_np, doc_ids_list_batch_for_np, formatted_docs_for_rag


    # --- OVERRIDE __call__ AND CORRECT target_device ---
    def __call__(
        self,
        question_input_ids: torch.LongTensor,
        question_hidden_states: Union[torch.FloatTensor, np.ndarray], # Can be tensor or ndarray
        prefix: Optional[str] = None,
        n_docs: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> ModelOutput:
        
        n_docs_to_use = n_docs if n_docs is not None else self.config.n_docs
        effective_return_tensors = return_tensors if return_tensors is not None else "pt"
        
        # --- CORRECTED target_device DETERMINATION ---
        if isinstance(question_hidden_states, torch.Tensor):
            target_device = question_hidden_states.device
        elif isinstance(question_input_ids, torch.Tensor): # Fallback if qhs is somehow not a tensor
            target_device = question_input_ids.device
        else:
            # This case should not be hit if called by RagModel, as it passes tensors.
            # Default to the device custom_retriever is on if inputs aren't tensors.
            target_device = self.custom_retriever.device
            # print(f"Warning: Inferring target_device from custom_retriever.device in HFCompatDenseRetriever.__call__")
        # --- END CORRECTION ---
        
        question_hidden_states_np: np.ndarray
        if isinstance(question_hidden_states, torch.Tensor):
            question_hidden_states_np = question_hidden_states.cpu().numpy()
        elif isinstance(question_hidden_states, np.ndarray):
            question_hidden_states_np = question_hidden_states
        else:
            raise TypeError(f"question_hidden_states must be torch.Tensor or np.ndarray, got {type(question_hidden_states)}")

        retrieved_doc_embeds_np, doc_ids_np, formatted_docs_textual = self.retrieve(
            question_hidden_states_np, n_docs_to_use
        )

        # Determine dtype for retrieved_doc_embeds_final based on input question_hidden_states if it was a tensor
        final_embed_dtype = torch.float
        if isinstance(question_hidden_states, torch.Tensor):
            final_embed_dtype = question_hidden_states.dtype
        
        retrieved_doc_embeds_final = torch.tensor(retrieved_doc_embeds_np, dtype=final_embed_dtype, device=target_device)
        doc_ids_final = torch.tensor(doc_ids_np, dtype=torch.long, device=target_device)

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        
        # postprocess_docs from parent RagRetriever is called. It uses self.generator_tokenizer.
        # It returns CPU tensors when return_tensors="pt".
        context_input_ids_cpu, context_attention_mask_cpu = self.postprocess_docs(
            formatted_docs_textual, input_strings,
            prefix if prefix is not None else self.config.generator.prefix,
            n_docs_to_use, return_tensors="pt",
        )

        context_input_ids_final = context_input_ids_cpu.to(target_device)
        context_attention_mask_final = context_attention_mask_cpu.to(target_device)
        
        return ModelOutput(
            context_input_ids=context_input_ids_final,
            context_attention_mask=context_attention_mask_final,
            retrieved_doc_embeds=retrieved_doc_embeds_final,
            doc_ids=doc_ids_final,
            docs=formatted_docs_textual
        )