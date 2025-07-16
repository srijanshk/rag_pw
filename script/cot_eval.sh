#!/usr/bin/env bash
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"  
BATCH=6                                        
MAX_NEW=768                                    
WPROJ="COT_EVAL"                            
STAMP=$(date +%Y%m%d_%H%M)

mkdir -p logs results

run_job () {
  local NAME=$1         
  local DATA=$2        
  echo "===== Running $NAME ====="
  accelerate launch run_zero_cot.py \
      --model_name   "$MODEL" \
      --dataset_path "$DATA" \
      --split        test \
      --out_path     results/${NAME}_COT_${STAMP}.jsonl \
      --batch        $BATCH \
      --max_new      $MAX_NEW \
      --wandb_project $WPROJ \
      --wandb_run_name ${NAME}_COT_llama3_${STAMP} \
  |& tee logs/${NAME}_COT_${STAMP}.log
  echo "===== $NAME done ====="
}

run_job gsm   data/benchmarks/gsm8k
run_job math  data/benchmarks/math

echo "✅ All datasets finished – results are in results/, logs in logs/"
