# Qwen3-4B DeltaMem Session adapter (LoCoMo 50.02%)

This directory contains the **trained DeltaMem adapter**, not the Qwen3-4B base weights. It is the adapter used for the 50.02% LoCoMo result under the `official_prompt` evaluation mode.

## Files

| File | Purpose |
|---|---|
| `delta_mem_adapter.pt` | Trained adapter weights (11,017,563 bytes) |
| `delta_mem_config.json` | Adapter architecture and hyperparameters |
| `training_summary.json` | Original training metadata from the GPU5 run |
| `SHA256SUMS` | Checksums of the three original files |

The required frozen backbone is `Qwen/Qwen3-4B-Instruct-2507`; its weights are distributed separately by Qwen.

## Provenance and evaluation

- Source run: `autocompressor_session_osgm_dd_decay_dmax2_qasper_history_turn_from_scratch_20260907` on GPU5.
- Training: 2,219 samples, seed 42, four GPUs, `autocompressor_session_episode`, segment length 500, maximum length 6,000, maximum write length 8,192, SDPA.
- Adapter: rank 8, alpha 16, Q/O attention heads, `online_gain=0.05`; see `delta_mem_config.json` for the full architecture.
- LoCoMo: 10 conversations, 1,540 questions, seed 42, session granularity, `full_history_mode=official_prompt`, batch size 32, maximum answer length 50. History decoding used sampling with temperature 0.4, top-p 0.9 and top-k 10.
- Result: Overall **50.02%**; multi-hop 40.25%, temporal 39.69%, open-domain 17.08%, single-hop 61.00%.
- Original GPU5 result SHA-256: `5344437a8abd5a3885f83f17e3f18fd57f7caf3e17bf7dd2b762e080b5b8af2a`. The original result file is not included here because this directory is for model artifacts.

The 46.69% run used `history_replay` and is **not** the evaluation protocol for this result. Its adapter weight and configuration checksums match these files; the evaluation mode differs.

The runtime used for the reported result includes the Session/OSGM dd_decay implementation from the GPU5 experiment. The repository's earlier public commit alone may not contain that full runtime. Treat this package as the verified model artifact and use the matching implementation and `official_prompt` settings when reproducing the score.
