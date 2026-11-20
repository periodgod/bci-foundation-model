<div align="center">

# bci-foundation-model

**A lightweight EEG research base: six encoder structures, four training objectives, three tuning strategies — all under a 5M parameter budget.**

轻量脑电基础模型研究底座：主候选约 **4.91M 参数**，在同一数据契约、同一划分协议下横向比较编码器结构、预训练目标与调优策略。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#快速开始--quick-start)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#快速开始--quick-start)
[![Params](https://img.shields.io/badge/main%20candidate-4.91M-orange?style=flat-square)](#结构--architectures)
[![Status](https://img.shields.io/badge/status-research%20base-yellow?style=flat-square)](#结果状态--results-status)

</div>

---

## 项目简介 / Overview

脑电数据规模远小于自然语言，直接把大模型架构搬过来会参数与数据错配。本项目在 **5M 参数预算**内做系统性对照：同一份数据契约下，换编码器结构、换预训练目标、换调优策略，看哪些设计真正带来增益。

结论先行：**工程链路成立，但"基础模型优于从头训练"在现有数据规模上没有成立。** 具体证据见下方结果表。

> A 5M-budget EEG research base. The engineering pipeline is verified end to end across 69 real-data runs; the claimed benefit of a foundation model over training from scratch is **not** established at the current data scale.

---

## 结构 / Architectures

代码实现 6 种编码器，切换 `model.architecture` 即可（[`src/eeg_mime_foundation/model.py`](src/eeg_mime_foundation/model.py)）：

| 结构 | 核心机制 | 对应设计意图 |
|---|---|---|
| `fusion`（主候选） | patch 线性嵌入 + 波形卷积分支 + FFT 频谱分支，三路融合 | 时域与频域信息同时进入 token |
| `factorized` | 先按通道做时间自注意力，再按时间片做空间自注意力 | 时空解耦，降低注意力复杂度 |
| `compact` | depthwise Conv1d 时间卷积 + 通道上下文混合 | 最小参数预算下的卷积替代 |
| `latent` | 可学习 latent query 压缩任意数量通道，再检索回通道维度 | **可变电极布局**：通道数变化时无需改结构 |
| `hierarchical` | 局部分块处理后接 epoch 级 GRU | 长序列分层建模 |
| `causal` | 时间维单向 GRU，无未来信息 | 在线/实时推理 |

所有结构共享：通道身份嵌入 + 3D 电极坐标嵌入 + 时间位置编码 + 零初始化残差 adapter + 三头输出（分类 / 重建 / 频带功率）。

### 关于 BSG 图展开

仓库历史中存在一次尝试（commit `35e4dea`，"Replace MI/ME encoder with balanced signed graph unrolling"），用平衡符号图展开替代注意力。**该实现不在当前训练管线中**——后续的数据管道、训练循环与检查点均基于上表 6 种结构。

对外表述时请注意区分：*设计过图展开方案* ≠ *当前模型使用图展开*。

---

## 训练目标 / Objectives

| 目标 | 说明 |
|---|---|
| `supervised` | 纯监督分类 |
| `masked` | 通道-时间联合掩码重建 + 频谱/频带约束 |
| `contrastive` | 双视图对比 |
| `ema` | EMA 自监督 + 监督头 |
| 蒸馏 | 单/多教师，等权与学习门控 |

调优策略三档：`full`（全量微调）、`linear`（冻结编码器线性探测）、`adapter`（只训残差 adapter）。

---

## 结果 / Results

**69 个已核验源码与检查点的真实数据运行**（运动想象 + 睡眠，各 10 名被试；主指标为逐被试平衡准确率再对种子平均，± 为三种子标准差）：

| 路径 | MI BA | 睡眠 BA |
|---|---:|---:|
| 融合结构从零监督 | **51.72% ± 2.98** | 67.17% ± 3.84 |
| 自研时空结构（取消融合） | 50.45% ± 0.45 | 57.98% ± 1.60 |
| 掩码预训练 → 全量微调 | 51.37% ± 1.22 | **69.18% ± 1.82** |
| 掩码预训练 → 冻结探测 | 53.68% ± 1.99 | 47.01% ± 4.61 |
| 掩码预训练 → adapter | 53.45% ± 1.81 | 47.14% ± 2.45 |

教师链路（仅 seed17）：EMA 预训练→微调 53.10/70.60；单教师 50.00–50.75；双教师等权 52.41/72.83；学习门控 51.39/65.99。
少标签支持集（每类 8 试次，三种子均值）：scratch 51.24 / 30.26，masked_full 54.27 / 51.47。
共享基座迁移（seed17）：full 50.43 / 68.04，linear 51.29 / 42.90。

### 结论边界

- 运动想象**接近机会水平**（二分类 50%），任何 MI 上的"提升"都在噪声量级内；
- 掩码预训练在睡眠任务上有小幅增益（67.17 → 69.18），但**未做被试级显著性检验**；
- 冻结探测在睡眠上**显著退化**（47.01），说明跨任务迁移依赖微调而非表征本身；
- 因此当前**不声称**通用基础模型收益成立。

完整验收记录：[`reports/基础模型直接研发验收.md`](reports/基础模型直接研发验收.md)。

---

## 结果状态 / Results Status

| 实验 | 状态 | 说明 |
|---|---|---|
| 通道丢弃鲁棒性（MI，小容量模型） | ✅ 有结果 | [`reports/channel_dropout_mi.json`](reports/channel_dropout_mi.json)：154k 参数模型基线 BA 53.6%，保留 8 通道降至 49.5%（20 次随机抽样）。注意小容量模型基线本身偏低 |
| 6 结构对照（MI，固定 40 epoch，seed 17） | ⚠️ 已完成，**差异不显著** | [`reports/arch_ablation_mi.json`](reports/arch_ablation_mi.json)：跨度 0.506–0.581，小于噪声量级。见下节 |
| 导联子集迁移（T1，仿真稀疏布局） | ⚠️ 已跑通，**结论不成立** | 64→32/19/8/3 导的 Δ 全部不显著；根因是基线本身处于机会水平。详见下节 |
| 大容量主候选的通道鲁棒性 | ⚠️ **无法给出结论** | 4.9M 主候选在 270 训练窗口下塌缩为常数预测（BA 0.5000），曲线无意义 |
| 真实跨设备 / 跨数据集迁移 | ❌ **无数据** | 需要同任务、不同硬件的第二份数据（2a / 2b / OpenBMI），当前只有单一 64 导布局 |
| BSG 图结构消融 | ❌ **不适用** | BSG 未进入当前管线（见上文） |
| 正式预训练 | ❌ **未启动** | 受限于数据规模与清洗协议确认 |
| EEG-to-LLM 指令调优 | ❌ **未实现** | 仓库内无相关代码 |

> 原则：**没有数据就写没有数据**，不用合成数据冒充结果。缺失项在上表中逐条列出。

### 结构对照（MI）：六个结构都停在机会水平附近

同一划分、同一 epoch 预算（40）、同一随机种子（17），只改 `model.architecture`：

| 结构 | 参数量 | BA |
|---|---|---|
| causal | 71k | 0.581 |
| latent | 187k | 0.559 |
| compact | 47k | 0.554 |
| fusion | 154k | 0.546 |
| factorized | 138k | 0.526 |
| hierarchical | 179k | 0.506 |

跨度 0.506–0.581 共 7.6 个百分点，而 135 个留出样本上单次评测的标准误约 4.3 个百分点——
**这个跨度落在噪声区间内，不能据此给结构排序**；且只跑了单一 seed，下结论需多种子重复。
唯一可以说的：在这份数据上，没有哪个结构明显跑出机会水平。

### 导联子集迁移（T1）：脚本可跑，但当前数据给不出有效结论

同一套权重，只把部分通道置为无效（电极 ID 与坐标保留），模拟换一顶更稀疏的电极帽。
135 个留出样本（4 名未见被试），指标取被试级 macro BA，Δ 用 2000 次配对 bootstrap 给 95% CI。
复现：`scripts/eval_montage_transfer.py`，结果：[`reports/montage_transfer_mi_fusion.json`](reports/montage_transfer_mi_fusion.json)。

| 条件 | 保留导联 | BA | Δ vs 全导联 | 95% CI |
|---|---|---|---|---|
| 全导联 | 64 | 0.551 | — | [0.469, 0.630] |
| 10-10 系统 | 32 | 0.525 | −0.026 | [−0.108, +0.051] |
| 10-20 系统 | 19 | 0.504 | −0.047 | [−0.127, +0.032] |
| 感觉运动区 | 8 | 0.522 | −0.029 | [−0.134, +0.077] |
| BCIV-2b 布局 | 3 | 0.525 | −0.027 | [−0.129, +0.066] |

**这张表必须读对。** 所有条件都没有显著下降，**不能**当作"模型对稀疏导联鲁棒"的证据——
基线本身 95% CI 覆盖 0.5，本来就没有可损失的性能，砍到 3 导自然也不掉。
同样的道理，脑区整块移除（中央区 −0.001、枕区 −0.002、额区 −0.059）同样全部不显著。

要让它变成有效结论，前提是基线先高于机会水平：引入 **BCI Competition IV 2a**
（22 导 / 9 被试 / 每被试 288 试次）把被试内解码做到 70% 以上，再测同一组导联子集。

与此配套，仓库验证了一个**结构性性质**而非分数：给 64 通道输入补 16 个标记为无效的通道，
logits 差异严格为 0（`tests/test_montage.py::test_padding_does_not_change_output`）。
也就是说"任意通道数可进"在数值上是真成立的，不是名义上的——缺的只是能体现差异的强基线。

---

## 快速开始 / Quick Start

```bash
pip install -e ".[eeg]"

# 结构消融（固定清单、固定 epoch、多种子）
python scripts/run_arch_ablation.py \
    --manifest <manifest.json> --base-config <config.json> \
    --output reports/arch_ablation_mi.json --seeds 17 --epochs 8

# 通道丢弃鲁棒性（需多通道且有信号的数据集）
python scripts/eval_channel_dropout.py \
    --checkpoint <best.pt> --manifest <manifest.json> \
    --output reports/channel_dropout.json --reps 10

# 导联子集迁移 T1（固定 10-20 / 10-10 / 3 导布局 + 随机下采样 + 脑区移除，带配对 bootstrap）
python scripts/eval_montage_transfer.py \
    --checkpoint experiments/outputs/compact-fusion/best.pt --manifest <mi manifest> \
    --output reports/montage_transfer_mi_fusion.json --reps 20

# 训练紧凑骨干（大容量模型在小样本上会塌缩，评测前先用这个拿到非退化权重）
python scripts/train_compact.py --name compact-fusion --arch fusion \
    --manifest <mi manifest> --base-config <config.json> --epochs 60

# 单次训练 / 评估
python -m eeg_mime_foundation.cli train --config configs/fusion_masked.json \
    --manifest <manifest.json> --output experiments/outputs/masked
python -m eeg_mime_foundation.cli evaluate --checkpoint experiments/outputs/masked/best.pt \
    --manifest <manifest.json> --split validation --output experiments/outputs/masked-eval
```

每个实验使用**新目录**，避免覆盖历史证据。

---

## 目录结构 / Layout

```
bci-foundation-model/
├── src/eeg_mime_foundation/   # 编码器、目标函数、训练循环、CLI
├── configs/                   # 配方（supervised / masked / contrastive / ema / 三档调优）
├── scripts/                   # 真实数据准备、消融、通道丢弃、蒸馏、few-shot、报告
├── docs/                      # 架构、路线图、协议、5M 改造方案
├── reports/                   # 验收记录与消融结果
├── tests/                     # 单元测试
└── experiments/ results/      # 本地输出（不入库）
```

## 文档 / Docs

- [`docs/architecture.md`](docs/architecture.md) — 编码器与数据流
- [`docs/5M_MI_ME模型改造方案.md`](docs/5M_MI_ME模型改造方案.md) — 参数预算与模块取舍
- [`docs/基础模型直接验证协议.md`](docs/基础模型直接验证协议.md) — 验证入口与门禁
- [`docs/训练前问题清单.md`](docs/训练前问题清单.md) — 启动正式预训练前的检查项
- [`docs/联合预训练与共享基座规范.md`](docs/联合预训练与共享基座规范.md) — MI + 睡眠共享基座

## 在项目群中的位置

训练与评测协议复用 [bci-bench](https://github.com/periodgod/bci-bench) 的数据契约、划分与指标定义；下游系统见 [bci-arm-robot](https://github.com/periodgod/bci-arm-robot) 与 [bci-visual-app](https://github.com/periodgod/bci-visual-app)。

---

<div align="center"><sub>Wuyi University · BCI Group　|　数据与权重不随仓库分发</sub></div>
