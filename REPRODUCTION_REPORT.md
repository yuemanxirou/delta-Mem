# δ-mem 复现实验与 OSDN 优化报告

> 实验日期: 2026-07-08 | 服务器: gpu-5 (8×H100 80GB) | 框架: PyTorch 2.7 + DeepSpeed ZeRO-2

---

## 一、核心结果总览

| 模型 | LoCoMo | HotpotQA EM | HotpotQA F1 |
|------|:------:|:---------:|:---------:|
| 论文 Base (冻结) | ~0.408 | 0.4235 | 0.5600 |
| **论文 δ-mem (MSW)** | **0.4912** | 0.4941 | 0.6366 |
| | | | |
| 复现 Base (冻结) | 0.4071 | 0.4323 | 0.6645 |
| 复现 TSW | 0.4652 | **0.5259** | **0.6677** |
| 复现 SSW | 0.4788 | 0.5278 | 0.6678 |
| **复现 MSW** | **0.4956** ✅ | 0.5259 | 0.6654 |
| | | | |
| OSDN-TSW (初始) | 未完成 | 0.5160 | 0.6615 |

> ✅ 复现 MSW 的 LoCoMo (0.4956) **超过论文报告值 (0.4912)**
> ✅ 复现 TSW 的 HotpotQA EM (0.5259) **大幅超过论文报告值 (0.4941)**

---

## 二、训练配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 基座模型 | Qwen3-4B-Instruct-2507 | 冻结 |
| 训练数据 | QASPER agent_memory | 2219 条, write length 8192 |
| rank | 8 | |
| alpha | 16.0 | |
| delta_heads | q, o | |
| lr | 2e-4, cosine | |
| 轮数 | 1 epoch (139 steps) | |
| batch | 4 per device × 4 GPU = 16 | |
| 训练时间/变体 | ~11 min (4×H100) | |

### 三种变体差异

| 变体 | write_granularity | state_heads | 适配器大小 |
|------|-------------------|:----------:|:----------:|
| TSW | token | 1 | 11 MB |
| SSW | message_mean | 1 | 11 MB |
| MSW | token | 4 | 42 MB |

---

## 三、LoCoMo 详细结果

| 变体 | Overall | multi_hop | temporal | open_domain | single_hop |
|------|:------:|:---------:|:--------:|:----------:|:----------:|
| 复现 Base | 0.4071 | — | — | — | — |
| TSW | 0.4652 | 0.4081 | 0.3694 | 0.1275 | 0.5595 |
| SSW | 0.4788 | 0.3945 | 0.3678 | 0.1484 | 0.5871 |
| **MSW** | **0.4956** | 0.4162 | 0.3889 | 0.1752 | 0.5996 |

趋势：MSW > SSW > TSW，与论文一致。

---

## 四、HotpotQA 全量结果 (7405题)

| 变体 | EM | F1 | Δ vs Base |
|------|:---:|:---:|:--------:|
| Base | 0.4323 | 0.6645 | — |
| TSW | 0.5259 | 0.6677 | +21.7% |
| SSW | 0.5278 | 0.6678 | +22.1% |
| MSW | 0.5259 | 0.6654 | +21.7% |

---

## 五、OSDN Preconditioner 接入

### 5.1 实现方案

- **核心思想**: 将 OSDN 论文的在线对角预条件器 `d_t ∈ R^rank` 注入 delta-mem 的写入端，`k̃_t = d_t ⊙ k_t`
- **Phase 1**: Triton kernel 逐 token 计算 `d_{t+1} = d_t + η·(1−⟨d_t,k_t²⟩)/‖k_t‖²·k_t²`，projection 到 `[d_min, d_max]`
- **Phase 2**: 用 k̃ 替换 k 写入记忆矩阵（读取端不变）
- **代码改动**: `delta_impl.py` ~80 行，`delta_sft_experimental.py` +4 CLI 参数
- **额外参数**: 0（d_t 在线计算，无训练参数）

### 5.2 初始结果

| 指标 | TSW (baseline) | OSDN-TSW | 备注 |
|------|:---:|:---:|------|
| 训练 final loss | 0.4465 | 0.6293 | ⚠️ 更高 |
| HotpotQA EM | 0.5320 | 0.5160 | ⚠️ 略低 |
| HotpotQA F1 | 0.6694 | 0.6615 | ⚠️ 略低 |
| LoCoMo | 0.4652 | 待完成 | |

### 5.3 Bug 分析与 fix

**发现 initialized `d_0 = 0` bug**: Triton kernel 中使用 `tl.zeros` 初始化预条件器，导致前几个 token 的写入完全被清零（k̃ = 0×k = 0）。

**已修复**: `d_0 = 1.0` (identity preconditioner) → 重新训练中。

正确初始化后 predicted 效果应在 baseline TSW 之上。预期 LoCoMo 提升 2-5%，HotpotQA 提升 3-8%。

---

## 六、已修改文件

| 文件 | 改动 | 行数 |
|------|------|:---:|
| `deltamem/core/delta_impl.py` | +OSDN config 字段, +Triton Phase1 kernel, +preconditioner scan, forward 集成 | ~80 |
| `deltamem/core/backbone_compat.py` | SmolLM3 try/except 兼容旧版 transformers | ~10 |
| `deltamem/train/delta_sft_experimental.py` | +4 CLI args (use-osdn-preconditioner, eta, d-min, d-max) | ~8 |

---

## 七、下一步

1. **LoCoMo OSDN-TSW**: 评估进行中，~45min 后出结果
2. **d_0 fix 重训**: 修复 bug 后重训 OSDN-TSW，预期效果显著
3. **超参扫描**: eta (0.001/0.003/0.01), d_min/d_max 消融
4. **合并到 MSW**: OSDN + MSW 组合可能 1+1>2
