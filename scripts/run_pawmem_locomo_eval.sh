#!/usr/bin/env bash
# Evaluate PAWmem under the official LoCoMo prompt protocol (50.02% run).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export DELTA_MEM_IMPL=ssw
export DELTA_MEM_SCAN_IMPL=triton_osgm_dd_decay
export DELTA_MEM_OSGM_DD_DECAY_DMIN=1.0
export DELTA_MEM_OSGM_DD_DECAY_DMAX=2.0
MODEL_PATH="${BASE_MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
ADAPTER_DIR="${PAWMEM_ADAPTER_DIR:-models/PAWmem_Qwen3_4B_QASPER_50p02}"
DATA_FILE="${PAWMEM_LOCOMO_FILE:-data/locomo10.json}"
OUTPUT_JSON="${PAWMEM_EVAL_OUTPUT:-outputs/pawmem_locomo_official_prompt.json}"
GPUS="${PAWMEM_GPUS:-4}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then echo "Missing local Qwen3-4B base model at $MODEL_PATH. See docs/PAWmem_reproduction.md" >&2; exit 2; fi
if [[ ! -f "$ADAPTER_DIR/delta_mem_adapter.pt" ]]; then echo "Missing PAWmem adapter: $ADAPTER_DIR" >&2; exit 2; fi
if [[ ! -f "$DATA_FILE" ]]; then echo "Missing LoCoMo data: $DATA_FILE" >&2; exit 2; fi
mkdir -p "$(dirname "$OUTPUT_JSON")"
python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
  -m deltamem.eval.locomo_delta \
  --model-path "$MODEL_PATH" --adapter-dir "$ADAPTER_DIR" --data-file "$DATA_FILE" \
  --device cuda:0 --dtype bfloat16 --attn-implementation sdpa \
  --skip-base --delta-conditions official_prompt --categories 1 2 3 4 \
  --seed 42 --eval-batch-size 32 --max-new-tokens 50 --answer-reserve-tokens 50 \
  --official-do-sample --official-temperature 0.4 --official-top-p 0.9 --official-top-k 10 \
  --output-json "$OUTPUT_JSON" "$@"
