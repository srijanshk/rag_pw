#!/usr/bin/env bash
set -euo pipefail

# -------- Model & Project --------
MODEL="meta-llama/Llama-3.1-8B-Instruct"
WPROJ="Dynamic_rag_EVAL_v2"
STAMP=$(date +%Y%m%d_%H%M)

# -------- Knowledge Bases (choose one or both) --------
KBS=("openmath" "mathpile")
# KBS=("mathpile")

# -------- Paths for each KB --------
OPENMATH_INDEX="/local00/student/shakya/openmath_bge-m3_hnsw_index"
OPENMATH_META="/local00/student/shakya/openmath_bge-m3_metadata.jsonl"

MATHPILE_INDEX="/local00/student/shakya/mathpile_hnsw.index"
MATHPILE_META="/local00/student/shakya/mathpile_meta.jsonl"

# -------- Benchmarks / Data --------
EXAMPLE_PATH="data/openmathinstruct2/example_id_to_data.json"
CHUNK_PATH="/local00/student/shakya/chunk_texts.json"

MODES=("summary")
# MODES=("raw" "summary")

# Default k_final (used for summary)
K_FINAL=5

# Full system settings
MAX_TOOL_CALLS=3

TOOL_TOK_GSM=512
ANS_TOK_GSM=2048
TOOL_TOK_MATH=512
ANS_TOK_MATH=2048

mkdir -p logs results

resolve_kb () {
  local KB="$1"
  case "$KB" in
    openmath)
      echo "$OPENMATH_INDEX|$OPENMATH_META"
      ;;
    mathpile)
      echo "$MATHPILE_INDEX|$MATHPILE_META"
      ;;
    *)
      echo "Unknown KB: $KB" >&2
      exit 1
      ;;
  esac
}

# -------- Single job runner --------
run_job () {
  local NAME="$1"     # e.g., gsm8k | math | math500
  local DATA="$2"     # dataset path or HF hub id
  local MODE="$3"     # summary | raw
  local KB="$4"       # openmath | mathpile

  # Per-benchmark token selection
  local TOOL_TOK=$TOOL_TOK_GSM
  local ANS_TOK=$ANS_TOK_GSM
  if [[ "$NAME" == "math" || "$NAME" == "math500" ]]; then
    TOOL_TOK=$TOOL_TOK_MATH
    ANS_TOK=$ANS_TOK_MATH
  fi

  # Resolve KB-specific FAISS files
  local RESOLVED; RESOLVED="$(resolve_kb "$KB")"
  local INDEX_PATH="${RESOLVED%%|*}"
  local METADATA_PATH="${RESOLVED##*|}"

  # Set k_final by mode: raw=3, summary=5
  local KFIN=$K_FINAL
  if [[ "$MODE" == "raw" ]]; then
    KFIN=3
  else
    KFIN=5
  fi

  local TAG="${NAME}_${KB}_${MODE}_${STAMP}_PROMPT"
  echo "===== Running $NAME | KB: $KB | mode: $MODE | k_final: $KFIN ====="
  python dynamic_rag.py \
      --benchmark         "$NAME" \
      --model_name        "$MODEL" \
      --dataset_path      "$DATA" \
      --faiss_index       "$INDEX_PATH" \
      --faiss_meta        "$METADATA_PATH" \
      --k_final           $KFIN \
      --max_tool_calls    $MAX_TOOL_CALLS \
      --tool_gen_tokens   $TOOL_TOK \
      --answer_gen_tokens $ANS_TOK \
      --injection_mode    "$MODE" \
      --quantize_4bit     False \
      --out_path          "results/dynamic_v2/${TAG}.json" \
      --wandb_project     "$WPROJ" \
      --wandb_run         "${NAME}__${KB}__${MODE}__FULL_SYS_${STAMP}" \
  |& tee "logs/${TAG}.log"
  echo "===== Done: $NAME | KB: $KB | mode: $MODE ====="
}

# -------- Sweep --------
for KB in "${KBS[@]}"; do
  for MODE in "${MODES[@]}"; do
    run_job gsm8k "data/benchmarks/gsm8k" "$MODE" "$KB"
    # run_job math data/benchmarks/math "$MODE" "$KB"
    run_job math500 "HuggingFaceH4/MATH-500" "$MODE" "$KB"
  done
done

echo "✅ Finished modes: ${MODES[*]} over KBs: ${KBS[*]} — see results/ & logs/"
