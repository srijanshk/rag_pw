# Retrieval-Augmented Generation (RAG) — Practical Work

Reimplementation of [Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) for open-domain QA on the Natural Questions (NQ) dataset.

The implementation extends the original paper with:
- **E5-large-v2** as the retriever backbone (replacing DPR)
- **FAISS HNSW** index for scalable approximate nearest-neighbor search (replacing flat exact search)
- **Hybrid retrieval**: dense (E5) + sparse (BM25 via Xapian) fused with Reciprocal Rank Fusion (RRF)

---

## Pipeline

```
Question → [E5 Query Encoder] + [BM25/Xapian]
                ↓                     ↓
         FAISS HNSW Search     Xapian Sparse Search
                ↓                     ↓
           Dense Docs           Sparse Docs
                └──── RRF Hybrid Merge ────┘
                              ↓
                  [BART-large Generator] → Answer
```

---

## Quick Start

### 1. Install prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
source .venv/bin/activate
```

### 2. Install Xapian (optional, for sparse/hybrid retrieval)

```bash
sudo apt update
sudo apt install xapian-tools libxapian-dev python3-xapian
# Verify:
xapian-config --version
python3 -c "import xapian; print(xapian.version_string())"
```

### 3. Clone the repository

```bash
git clone https://github.com/srijanshk/rag_pw.git
cd RAG_PW
uv sync
source .venv/bin/activate
```

---

## Data Download

Follow the [DPR repository](https://github.com/facebookresearch/DPR) to download NQ data and the Wikipedia dump.

```bash
# QA pairs
python utils/download_data.py --resource data.retriever.qas
python utils/download_data.py --resource data.retriever.nq

# Wikipedia dump
python utils/download_data.py --resource data.wikipedia_split

# Gold question-passage pairs
python utils/download_data.py --resource data.gold_passages_info
```

---

## Workflow

### Step 1 — Fine-tune retriever and generator independently (optional warm-start)

```bash
python misc/train_retriever.py
python misc/train_generator.py
```

### Step 2 — Build the FAISS HNSW index from Wikipedia

```bash
python document_vector_index.py
```

### Step 3 — Build the Xapian (BM25) index from Wikipedia

```bash
python generate_xapian_index.py
```

### Step 4 — Pre-compute sparse retrieval results

```bash
python misc/sparse_retrieve_contexts.py
```

### Step 5 — Run end-to-end RAG training

```bash
python main.py
```

### Evaluation only

```bash
python run_eval.py
```

### Tests

```bash
python test_function.py
python bootstrap_testing.py
```

---

## Repository Structure

```
RAG_PW/
├── main.py                      # Training entry point; all hyperparameters set here
├── run_eval.py                  # Evaluation-only entry point
├── QuestionEncoder.py           # E5 wrapper with mean pooling (PreTrainedModel subclass)
├── DenseRetriever.py            # FAISS HNSW index wrapper + document encoder
├── RagUtils.py                  # Retrieval, hybrid merge, generator input prep, RAG loss
├── RagEval.py                   # Evaluation with EM/F1; RAG-Sequence decoding
├── NqDataset.py                 # PyTorch Dataset for NQ; tokenizes questions and answers
├── xapian_retriever.py          # Xapian BM25/TF-IDF wrapper
├── document_vector_index.py     # Builds FAISS HNSW index from Wikipedia passages
├── generate_xapian_index.py     # Builds Xapian sparse index from Wikipedia passages
├── utils.py                     # Data loading, custom collate function
├── utils/
│   └── download_data.py         # DPR-style data downloader
├── misc/
│   ├── train_retriever.py       # Stand-alone retriever fine-tuning
│   ├── train_generator.py       # Stand-alone generator fine-tuning
│   └── sparse_retrieve_contexts.py  # Pre-computes sparse retrieval JSONL files
├── pyproject.toml               # uv/pip dependency spec
└── uv.lock                      # Locked dependency versions
```

---

## Monitoring

Training uses [Weights & Biases](https://wandb.ai) for loss and eval metric logging. Run `wandb login` before starting, or set `WANDB_MODE=disabled` to skip it.

---

## Key Design Decisions vs. Original Paper

| Aspect | Lewis et al. (2020) | This Implementation |
|---|---|---|
| Retriever | DPR (BERT bi-encoder) | E5-large-v2 |
| Retrieval index | FAISS flat (exact MIPS) | FAISS HNSW (approximate) |
| Retrieval strategy | Dense only | Dense + BM25 hybrid (RRF) |
| Training | Query encoder + BART jointly; doc encoder frozen | Same |
| Loss | RAG-Sequence marginal NLL | Same |
| Evaluation decoding | RAG-Sequence: per-doc generation + marginalization | Same (`log P(doc|q) + log P(y|q,doc)`) |
