import faiss
import json
import numpy as np
import logging
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

class BGERetriever:
    def __init__(self,
                 embedding_model: BGEM3FlagModel,
                 index_path: str,
                 metadata_path: str,
                 device: str = 'cuda',
                 ef_search: int = 700
                 ):
        """
        Initializes the BGERetriever with a pre-trained embedding model, a Faiss index, and metadata.
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
        inner = self.index
        while hasattr(inner, "index"):
            inner = inner.index
        # ensure we have the proper subclass wrapper for HNSW
        inner = faiss.downcast_index(inner)

        # ensure it's truly an HNSW index
        if not hasattr(inner, "hnsw"):
            raise ValueError(f"Index at {index_path} isn’t an HNSW index (found {type(inner)})")

        # tune HNSW parameters
        inner.hnsw.efSearch = ef_search
        logger.info(f"Using HNSW index; set efSearch={ef_search}")


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
                    
        embedding_dict = self.model.encode(
            queries, 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False,
            batch_size=128
        )
        query_embeddings = embedding_dict['dense_vecs']
        query_embedding_32 = query_embeddings.astype(np.float32)

        
        # L2 normalize the query embedding for cosine similarity search
        faiss.normalize_L2(query_embedding_32)

        # Search the Faiss index
        distances, indices = self.index.search(query_embedding_32, k)
        
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