### RAG_PW

## Getting Started

Follow the detailed instructions in the sections below to set up the project and execute the workflow. Ensure all prerequisites are met before proceeding with the installation and data processing steps.



## Quick Installation Guide

1. **Install Prerequisites**
   - Ensure **Anaconda** or **Miniconda** is installed. Download from [Anaconda's official website](https://www.anaconda.com/products/distribution).

2. **Clone the Repository**
   ```bash
   git clone https://github.com/srijanshk/rag_pw.git
   cd RAG_PW
   ```

3. **Install Xapian(Optional but recommended)**
   1. **Update the Package List**
      ```sh
      sudo apt update
      ```

   2. **Install Xapian Core**
      ```sh
      sudo apt install xapian-tools libxapian-dev
      ```

   3. **Verify Installation**
      ```sh
      xapian-config --version
      ```

   4. **Optional: Install Python Bindings**
      If you need Python bindings for Xapian:
      ```sh
      sudo apt install python3-xapian
      ```

   5. **Test Installation**
      Try importing Xapian in Python:
      ```python
      import xapian
      print(xapian.version_string())
      ```

      If the version is printed, the installation was successful.


4. **Create the Environment**
   ```bash
   conda env create -f environment.yml
   ```

5. **Activate the Environment**
   ```bash
   conda activate srijan_pw_env
   ```

6. **Deactivate Environment (When Done)**
   ```bash
   conda deactivate
   ```

### Download data

Follow [DPR repo](https://github.com/facebookresearch/DPR.git) in order to download NQ data and Wikipedia DB. 

1. Download QA pairs by `python utils/download_data.py --resource data.retriever.qas` and `python3 data/download_data.py --resource data.retriever.nq`.
2. Download wikipedia DB by `python utils/download_data.py --resource data.wikipedia_split`.
3. Download gold question-passage pairs by `python utils/download_data.py --resource data.gold_passages_info`.
4. To make the subset of the data `python filter_subset_wiki.py --db_path downloads/data/wikipedia_split/psgs_w100.tsv --data_path downloads/data/retriever/nq-train.json`

