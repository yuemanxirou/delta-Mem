# PAWmem: reproduce QASPER training and LoCoMo evaluation

This is the complete runnable recipe for the **PAWmem Qwen3-4B × QASPER History-turn** checkpoint. The verified LoCoMo result is **50.02% (1,540 questions)** with `full_history_mode=official_prompt`. The 46.69% result used `history_replay` and is a different evaluation protocol. The JSON result key is named `full_history_replay` in both cases; check `full_history_mode` in the run metadata before quoting a score.

## 1. Clone and install

```bash
git clone https://github.com/yuemanxirou/delta-Mem.git
cd delta-Mem
bash scripts/setup_uv_env.sh
source .venv/bin/activate
```

Use Linux with NVIDIA CUDA and a CUDA-compatible PyTorch. The setup script installs the dependencies in `requirements.txt`, including Transformers, Datasets, DeepSpeed, and Triton. The recorded training and evaluation used **four GPUs** with bf16 and SDPA. Allow at least 20 GB of free output space for the training checkpoint (the one-step smoke run wrote about 16 GB before cleanup). The GPU5 smoke checks used Python 3.12, PyTorch 2.9.0, Transformers 5.14.1, Datasets 5.0.0, Accelerate 1.14.0, DeepSpeed 0.19.2, and Triton 3.2.0. The repository requirements are not a frozen lockfile. The frozen base model is [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507); download it with Hugging Face access, or set `BASE_MODEL_PATH` to a local copy. The base weights are distributed by Qwen and are not in this repository. Both scripts require a **local model directory**, because the model loader uses local files only. Download once:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
```

## 2. What to use

| Purpose | Repository file | SHA-256 |
| --- | --- | --- |
| Training data (2,219 QASPER History-turn examples) | `data/qasper_history_turn_sessions_20260907.jsonl` | `f2ee6d9f372f86117c03e978e86688b7619af1794d1ebdec90e6f41f459c21a7` |
| LoCoMo evaluation data (10 conversations, 1,540 non-adversarial questions) | `data/locomo10.json` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| Trained PAWmem adapter | `models/PAWmem_Qwen3_4B_QASPER_50p02/` | See `SHA256SUMS` there |
| Training implementation | `deltamem/train/delta_sft_experimental.py` | GPU5 Session/OSGM dd_decay version |
| Evaluation implementation | `deltamem/eval/locomo_delta.py` | GPU5 `official_prompt` version |
| Memory runtime and scan | `deltamem/core/`, `deltamem/runtime/`, `deltamem/kernels/` | See the PAWmem release commit |

The training JSONL is already included in GitHub. Each row has a `messages` conversation; one original QASPER History turn is represented as one session. The LoCoMo JSON is also included. No separate experiment dataset download is required.

Check the files after cloning:

```bash
sha256sum data/qasper_history_turn_sessions_20260907.jsonl data/locomo10.json
(cd models/PAWmem_Qwen3_4B_QASPER_50p02 && sha256sum -c SHA256SUMS)
```

## 3. Evaluate the released adapter

```bash
export BASE_MODEL_PATH=models/Qwen3-4B-Instruct-2507
export PAWMEM_GPUS=4
bash scripts/run_pawmem_locomo_eval.sh
```

The output is `outputs/pawmem_locomo_official_prompt.json`. The command loads the repository's adapter and LoCoMo data, and fixes the Session/OSGM data-dependent-decay runtime (`dmin=1`, `dmax=2`), 4 categories, seed 42, batch 32, 50 generation tokens, sampling parameters, and `official_prompt`. It evaluates the adapter only. To use a local backbone: `BASE_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 bash scripts/run_pawmem_locomo_eval.sh`.

For a short smoke run, append `--max-conversations 1 --max-questions-per-conversation 1` to the command. This checks execution only and cannot reproduce the 50.02% aggregate.

## 4. Train from scratch, then evaluate your checkpoint

```bash
export BASE_MODEL_PATH=models/Qwen3-4B-Instruct-2507
export PAWMEM_GPUS=4
bash scripts/run_pawmem_qasper_train.sh
```

Training writes to `outputs/pawmem_qasper_train` by default. It uses `autocompressor_session_episode`, 500-token segments, maximum 6,000-token document, 8,192-token write window, Q/O rank-8 alpha-16 adapter, one epoch, bf16/SDPA, four GPUs, per-GPU microbatch 1, gradient accumulation 4, cosine learning-rate schedule, 14-step warmup (10%), AdamW fused, seed 42, and DeepSpeed ZeRO-2. This is the configuration recovered from the original GPU5 `training_summary.json`, adapter config, training log, and trainer state. The original run completed 139 optimizer steps.

Evaluate the new adapter:

```bash
PAWMEM_ADAPTER_DIR=outputs/pawmem_qasper_train \
PAWMEM_EVAL_OUTPUT=outputs/pawmem_locomo_retrained.json \
bash scripts/run_pawmem_locomo_eval.sh
```

The original training script invocation was not archived verbatim. The commands above reconstruct its settings from the saved artifacts and logs. Exact retraining weights can differ across GPU/software versions; the included adapter is the verified 50.02% checkpoint. Training is resource intensive (the original four-GPU run took hours).

## 5. Recorded result

The [original GPU5 LoCoMo result JSON](results/PAWmem_QASPER_LoCoMo_official_prompt_50p02.json) is included and has SHA-256 `5344437a8abd5a3885f83f17f3e18fd57f7caf3e17bf7dd2b762e080b5b8af2a`, with overall 0.5002, multi-hop 0.4025, temporal 0.3969, open-domain 0.1708, single-hop 0.6100. Those figures correspond to the QASPER-trained PAWmem row in [the benchmark table](results/PAWmem_training_set_benchmark_table_2026-10-22.md). Other cells in that table are a preserved result record and are not produced by the two commands above.
