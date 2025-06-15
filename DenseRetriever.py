import faiss
import json
import numpy as np
import logging
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self,
                 embedding_model: BGEM3FlagModel,
                 index_path: str,
                 metadata_path: str,
                 device: str = 'cuda'):
        """
        A simplified retriever for Faiss indexes that uses a pre-initialized BGE-M3 model.

        Args:
            embedding_model: An already initialized BGEM3FlagModel instance.
            index_path: Path to the serialized Faiss index.
            metadata_path: Path to the JSONL metadata file.
            device: The device to use for encoding ('cuda' or 'cpu').
        """
        logger.info("Initializing DenseRetriever...")
        self.model = embedding_model
        self.device = device
        
        logger.info(f"Loading Faiss index from: {index_path}")
        self.index = faiss.read_index(index_path)
        # For GPU-based search, you can move the index to a GPU resource
        if self.device == 'cuda':
            try:
                self.res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index) # Use GPU 0
                logger.info("Successfully moved Faiss index to GPU.")
            except Exception as e:
                logger.warning(f"Could not move Faiss index to GPU, will use CPU for search. Error: {e}")

        logger.info(f"Loading metadata from: {metadata_path}")
        self.metadata = self._load_metadata(metadata_path)
        logger.info(f"✅ DenseRetriever ready. Loaded {len(self.metadata)} metadata entries.")

    def _load_metadata(self, path: str) -> dict:
        """Loads metadata from a JSONL file into a dictionary, keyed by integer ID."""
        meta = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # The 'id' in the metadata should match the integer ID in the Faiss index
                    meta[int(entry["id"])] = entry
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed metadata line: {line.strip()} - Error: {e}")
        return meta

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[dict]]:
        """
        Encodes a batch of queries and retrieves the top-k most relevant documents for each.
        """
        if not queries:
            return []
            
        # BGE-M3 does not require special prefixes like "query: "
        embedding_dict = self.model.encode(
            queries, 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False,
            batch_size=128 # A reasonable inference batch size
        )
        query_embeddings = embedding_dict['dense_vecs']
        
        # L2 normalize the query embedding for cosine similarity search
        faiss.normalize_L2(query_embeddings)

        # Search the Faiss index
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        batch_results = []
        for i in range(len(queries)):
            query_results = []
            for j in range(k):
                idx = indices[i][j]
                if idx == -1: continue # Faiss returns -1 if no more results
                
                metadata_entry = self.metadata.get(idx, {})
                query_results.append({
                    "id": idx,
                    "score": distances[i][j],
                    "solution_chunk": metadata_entry.get("solution_chunk", ""),
                    "problem": metadata_entry.get("problem", "")
                })
            batch_results.append(query_results)
            
        return batch_results

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Convenience method to search for a single query."""
        return self.search_batch([query], k)[0]