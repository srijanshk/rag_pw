#!/usr/bin/env bash
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"  
BATCH=2                                        
MAX_NEW=1024                                    
WPROJ="DYNAMIC_COT_EVAL"                            
STAMP=$(date +%Y%m%d_%H%M)
INDEX_PATH="/local00/student/shakya/openmath_bge-m3_hnsw_index"
METADATA_PATH="/local00/student/shakya/openmath_bge-m3_metadata.jsonl"
K_FINAL=5


mkdir -p logs results

run_job () {
  local NAME=$1         
  local DATA=$2        
  echo "===== Running $NAME ====="
  python dynamic_retrieval.py \
      --model_name   "$MODEL" \
      --dataset_path "$DATA" \
      --faiss_index   "$INDEX_PATH" \
      --faiss_meta "$METADATA_PATH" \
      --k_final      $K_FINAL \
      --split        test \
      --seed         1234 \
      --out_path     results/${NAME}_DYNAMIC_COT_${STAMP}.jsonl \
      --batch        $BATCH \
      --max_new      $MAX_NEW \
      --wandb_project $WPROJ \
      --wandb_run_name ${NAME}_COT_llama3_${STAMP} \
  |& tee logs/${NAME}_COT_${STAMP}.log
  echo "===== $NAME done ====="
}

# run_job math  data/benchmarks/math
run_job gsm   data/benchmarks/gsm8k

echo "✅ All datasets finished – results are in results/, logs in logs/"
