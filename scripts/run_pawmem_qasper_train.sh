#!/usr/bin/env bash
# Reproduce the PAWmem QASPER History-turn training configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export DELTA_MEM_IMPL=ssw
export DELTA_MEM_SCAN_IMPL=triton_osgm_dd_decay
export DELTA_MEM_OSGM_DD_DECAY_DMIN=1.0
export DELTA_MEM_OSGM_DD_DECAY_DMAX=2.0
MODEL_PATH="${BASE_MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${PAWMEM_TRAIN_FILE:-data/qasper_history_turn_sessions_20260907.jsonl}"
OUTPUT_DIR="${PAWMEM_OUTPUT_DIR:-outputs/pawmem_qasper_train}"
GPUS="${PAWMEM_GPUS:-4}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then echo "Missing local Qwen3-4B base model at $MODEL_PATH. See docs/PAWmem_reproduction.md" >&2; exit 2; fi
if [[ ! -f "$TRAIN_FILE" ]]; then echo "Missing training data: $TRAIN_FILE" >&2; exit 2; fi
python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
  -m deltamem.train.delta_sft_experimental \
  --model-path "$MODEL_PATH" --train-file "$TRAIN_FILE" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --bf16 --attn-implementation sdpa \
  --rank 8 --alpha 16 --delta-heads q,o --output-init base_slice_fixed \
  --base-slice-ref-width 8 --online-gain 0.05 \
  --memory-write-source learned_hidden --memory-write-granularity message_mean \
  --training-mode autocompressor_session_episode \
  --autocompressor-segment-length 500 --autocompressor-max-length 6000 \
  --autocompressor-loss-weight 0.1 --assistant-loss-mode all_assistant_turns \
  --episode-recent-messages 1 --max-write-length 8192 --max-length 6000 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 --num-train-epochs 1 --seed 42 --data-seed 42 \
  --lr-scheduler-type cosine --warmup-ratio 0.1 --optim adamw_torch_fused \
  --save-steps 100 --tf32 --deepspeed-config deepspeed_zero2.json \
  --write-sparsity-weight 0.01 --write-sparsity-target 0.05 \
  --dataset-num-proc 16 --dataloader-num-workers 8 \
  "$@"
