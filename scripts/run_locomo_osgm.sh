#!/bin/bash

set -x


_find_checkpoint() {
  local trainer_dir="$1"
  local latest=""
  if [[ -d "${trainer_dir}" ]]; then
    latest=$(find "${trainer_dir}" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -t- -k2 -n | tail -1)
  fi
  printf "%s" "${latest}"
}

cd /DATA/disk0/liuyuerou/projects/delta-Mem

source .venv/bin/activate


GPU_IDS="${GPU_IDS:-0,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# enable OSGM kernel
export DELTA_MEM_SCAN_IMPL=triton_osgm


python -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr 127.0.0.1 \
    --master_port 29700 \
    -m deltamem.eval.locomo_delta \
    --model-path \
    /DATA/disk0/liuyuerou/projects/delta-Mem/huggingface/Qwen3-4B-Instruct-2507 \
    --adapter-dir \
    $(_find_checkpoint "/DATA/disk0/liuyuerou/projects/delta-Mem/models/qwen3_4b_instruct_delta_mem_qasper_6k_seed42_rank8_qo_TSW_write8192_consistent70/trainer") \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --max-new-tokens 50 \
    --seed 42 \
    --eval-batch-size 32 \
    --answer-reserve-tokens 50 \
    --skip-base \
    --delta-conditions full_history_replay \
    --full-history-mode official_prompt \
    --categories 1 2 3 4 \
    --output-json \
    outputs/locomo_osgm_checkpoint278.json \
    --data-file \
    data/locomo10.json