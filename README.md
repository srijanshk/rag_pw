### RAG_PW
## Environment Setup

## Quick Installation Guide

1. **Install Prerequisites**
   - Ensure **Anaconda** or **Miniconda** is installed. Download from [Anaconda's official website](https://www.anaconda.com/products/distribution).

2. **Clone the Repository**
   ```bash
   git clone https://github.com/lizalengyel/RAG_PW.git
   cd RAG_PW
   ```

3. **Create the Environment**
   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the Environment**
   ```bash
   conda activate thesis_env
   ```

5. **Launch JupyterLab (Optional)**
   ```bash
   jupyter lab
   ```

6. **Deactivate Environment (When Done)**
   ```bash
   conda deactivate
   ```

---
## Quick Installation Guide for Xapian on Macbook

To install **Xapian** on a MacBook with an **M1/M2 chip** (Apple Silicon), follow these steps:

---

### **Step 1: Install Homebrew (if not already installed)**

1. Open **Terminal** and run:

   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. After installation, update Homebrew:

   ```sh
   brew update
   ```

---

### **Step 2: Install Xapian**
1. Use Homebrew to install Xapian:

   ```sh
   brew install xapian
   ```

2. To verify the installation, check the version:

   ```sh
   xapian-config --version
   ```

---

### **Step 3: Install Xapian Bindings (Optional)**
If you need **Python bindings** for Xapian:

1. First, install `swig` (needed for bindings):

   ```sh
   brew install swig
   ```

2. Then install Xapian bindings for Python:

   ```sh
   pip install xapian-bindings
   ```

---

### **Step 4: Test Installation**
Try importing Xapian in Python:

```python
import xapian
print(xapian.version_string())
```

If you see the Xapian version printed, the installation was successful.

---

### **Troubleshooting**
- If you face issues, ensure that Homebrew is installed correctly with:
  
  ```sh
  brew doctor
  ```

- If using **Apple Silicon (M1/M2/M4)**, ensure you are running the terminal in native ARM64 mode, not Rosetta 2 (x86 emulation).

---

### Download data

Follow [DPR repo][https://github.com/facebookresearch/DPR.git] in order to download NQ data and Wikipedia DB. 

1. Download QA pairs by `python utils/download_data.py --resource data.retriever.qas` and `python3 data/download_data.py --resource data.retriever.nq`.
2. Download wikipedia DB by `python utils/download_data.py --resource data.wikipedia_split`.
3. Download gold question-passage pairs by `python utils/download_data.py --resource data.gold_passages_info`.
4. To make the subset of the data `python filter_subset_wiki.py --db_path downloads/data/wikipedia_split/psgs_w100.tsv --data_path downloads/data/retriever/nq-train.json`

The script will create a new passage DB containing passages which originated articles are those paired with question on the original NQ data (78,050 unique articles; 1,642,855 unique passages).
This new DB will be stored at `downloads/data/wikipedia_split/psgs_w100_subset.tsv`.


### Wikipedia Dump, Chunk Creation, Xapian DB and Generate Fassis Index

## Overview

This guide explains the full workflow to process a Wikipedia dump, split it into smaller chunks, index the data using Xapian, and finally generate a Fassis index for efficient search and retrieval. The process is broken down into four main steps:

1. **Wikipedia Dump**: Download and prepare the Wikipedia dump.
2. **Chunk Creation**: Parse and split the large dump into manageable chunks.
3. **Xapian DB Creation**: Build a search index database from the chunks using Xapian.
4. **Generate Fassis Index**: Process the chunks and create a specialized Fassis index.

---

## 1. Wikipedia Dump

### Description

A Wikipedia dump is a large XML file (often compressed) containing the full content of Wikipedia articles. This file serves as the raw input for the indexing process.

### Steps

1. **Download the Dump:**
   - Visit [Wikipedia Dumps](https://dumps.wikimedia.org/enwiki/latest/) and download the latest version (e.g., `enwiki-latest-pages-articles.xml.bz2`).
2. **Process Dump with Wikiextractor:**
For Wikiextractor use this version from their repo:
https://github.com/attardi/wikiextractor/pull/313
```
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml --json --no-template -o output_dir
```

---

## 2. Chunk Creation and Xapian DB Creation

### Description

Due to the enormous size of the Wikipedia dump, it is necessary to split it into smaller, more manageable chunks. Each chunk may contain a fixed number of articles or be based on file size.
And The next step is to build a search-friendly database using Xapian. This involves reading each chunk file and indexing its contents

```
python generate_chunk_xapian.py
```
---

## 3. Generate Fassis Index

### Description

The final step involves generating a Fassis index from the chunks. 
```
python generate_encoding.py
```