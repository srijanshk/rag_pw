import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import argparse
import json
import torch
from retriever import DenseRetriever

def load_samples(path):
    """
    Load NQ-style JSON or JSONL files with 'question', 'context', and 'short_answers'.
    Returns list of dicts with 'query' and 'positive_context'.
    """
    samples = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                if obj.get('short_answers'):
                    samples.append({'query': obj['question'], 'positive_context': obj['context']})
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f).get('data', [])
            for obj in data:
                if obj.get('short_answers'):
                    samples.append({'query': obj['question'], 'positive_context': obj['context']})
    return samples


def train_overfit(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Initialize retriever with E5, fine-tuning enabled
    retriever = DenseRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        device=device,
        model_name=args.model_name,
        fine_tune=True,
        use_fp16=False,
        ef_search=args.ef_search,
        ef_construction=args.ef_construction
    )
    # override optimizer lr if provided
    retriever.optimizer = torch.optim.AdamW(
        retriever.query_encoder.parameters(), lr=args.lr
    )

    # Load and overfit on the first sample
    samples = load_samples(args.data_path)
    if not samples:
        raise ValueError(f"No samples loaded from {args.data_path}")
    sample = samples[0]
    batch = [{
        'query': sample['query'],
        'positive_doc': sample['positive_context'],
        'negative_docs': []
    }]

    print("=== Starting overfit retriever run on single sample ===")
    for epoch in range(1, args.epochs+1):
        loss = retriever.fine_tune_on_batch(batch)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}")

    # Evaluate final cosine similarity
    retriever.query_encoder.eval()
    embeddings_q = retriever.embed_texts_for_training(
        [sample['query']], "query", retriever.query_encoder, is_encoder_trainable=True 
    )
    embeddings_c = retriever.embed_texts_for_training(
        [sample['positive_context']], "passage", retriever.doc_encoder, is_encoder_trainable=True 
    )
    cos_sim = torch.nn.functional.cosine_similarity(embeddings_q, embeddings_c).item()
    print(f"Final cosine similarity on overfit sample: {cos_sim:.4f}")

    # Optional: perform a search to see if it's top-1
    results = retriever.search(sample['query'], k=5)
    print("Top-5 retrieval results:")
    for rank, res in enumerate(results, start=1):
        print(f"{rank}. id={res['id']} score={res['score']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path',    type=str, required=True, help='FAISS index file path')
    parser.add_argument('--metadata_path', type=str, required=True, help='Metadata JSONL with passages')
    parser.add_argument('--data_path',     type=str, required=True, help='NQ train/dev JSON or JSONL')
    parser.add_argument('--model_name',    type=str, default='intfloat/e5-large-v2', help='SentenceTransformer model')
    parser.add_argument('--device',        type=str, default='cuda:0', help='PyTorch device')
    parser.add_argument('--lr',            type=float, default=2e-5, help='Learning rate for retriever')
    parser.add_argument('--epochs',        type=int, default=20, help='Number of epochs to overfit')
    parser.add_argument('--ef_search',     type=int, default=1500, help='HNSW efSearch')
    parser.add_argument('--ef_construction',type=int, default=200, help='HNSW efConstruction')
    args = parser.parse_args()

    train_overfit(args)
