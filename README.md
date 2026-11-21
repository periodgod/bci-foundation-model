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

### 坐标与图偏置

坐标不再恒为零：`montage.py` 用 MNE `standard_1020` 模板把导联名解析成头表米制坐标
（EEGMMIDB 64 导 **64/64** 解析成功，双极导联取中点）。此前预处理写的是
`coords=0, coord_valid=False`，坐标分支实际上从未被激活——这一条已修正，
证据见 [`docs/全量语料与坐标协议.md`](docs/全量语料与坐标协议.md)。

在此之上新增可选的**图条件注意力**（`model.graph=true`）：成对电极距离经高斯核展开、
线性映射为每头加性偏置，零初始化注入空间注意力。它对应一个已知缺陷——
结构化移除 8 导（0.5288）与随机移除 8 导（0.5390）几乎等价，说明模型此前没有真正利用电极拓扑。
配套的**导联丢弃增广**（`training.channel_dropout`）在训练期整列屏蔽通道，逼模型适应更稀疏的帽子。

两者目前均为 `已实现、未验证`：单测覆盖了填充不变、通道置换等变、零初始化等价，
但性能对照尚未跑（协议见下文）。

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
| 6 结构对照（MI） | ⚠️ **未产出可用结果** | 早期一轮基于单次划分，结果已撤下；需在 LOSO 协议下重跑（见下节） |
| 跨被试留一（LOSO，fusion） | ✅ 有结果 | 10 折均值 **BA 0.660 ± 0.049**，范围 0.580–0.729。这是当前最可信的基线 |
| 导联子集迁移（T1，LOSO 协议） | ✅ 有结果 | 64→32/19/8/3 导单调下降，8 导与 3 导 **10/10 折全部下降**。见下节 |
| 大容量主候选的通道鲁棒性 | ⚠️ **无法给出结论** | 4.9M 主候选在 270 训练窗口下未能收敛到可用状态，通道曲线无意义 |
| 真实跨设备 / 跨数据集迁移 | 🟡 **数据与接口就绪，实验未跑** | 2a 2 592 窗口（5 导）、2b 6 520 窗口（3 导）已入册；三库共享词表 + 混批前向自检通过。**无迁移性能数字** |
| 全量语料（MI 109 被试 / 睡眠 100 被试） | ✅ **两侧均已切窗入库**：MI 4 917 窗口、睡眠 59 546 窗口；对照组未跑 | 见 [`docs/数据资产与缺口.md`](docs/数据资产与缺口.md) 与 [`docs/全量语料与坐标协议.md`](docs/全量语料与坐标协议.md) |
| 图偏置 / 导联丢弃对照 | 🟡 **代码与单测完成，结果待跑** | `GraphBlock`、`drop_channels`、`scripts/run_grouped_cv.py` 均已就绪 |
| BSG 图结构消融 | ❌ **不适用** | BSG 未进入当前管线（见上文） |
| 正式预训练 | ❌ **未启动** | 受限于数据规模与清洗协议确认 |
| EEG-to-LLM 指令调优 | ❌ **未实现** | 仓库内无相关代码 |

> 原则：**没有数据就写没有数据**，不用合成数据冒充结果。缺失项在上表中逐条列出。

### 跨被试留一（LOSO）：当前最可信的基线

10 折，每折留一名被试做验证、其余被试训练，fusion 结构、40 epoch：

| 指标 | 值 |
|---|---|
| 10 折均值 BA | **0.6595** |
| 折间标准差 | 0.0488 |
| 折间范围 | 0.580 – 0.729 |
| 参数量 | 154k |

复现：`scripts/run_loso.py`，结果：[`experiments/outputs/loso-fusion/summary.json`](experiments/outputs/loso-fusion/summary.json)。

这个数字比早期单次划分高出约 11 个百分点，说明**单次留出划分在本数据规模下不可靠**：
同样权重换个划分就能从 0.55 跳到 0.66。因此本仓库后续所有对照一律走 LOSO，不再用单次划分下结论。

### 结构对照（MI）：尚未产出可用结果，待在 LOSO 协议下重跑

早期做过一轮六结构对照（同一划分、40 epoch、seed 17），但这批实验跑在**后来被证明不具代表性的单次划分**上：
同一结构（fusion）在 LOSO 下是 0.660，在那次划分上只有 0.55，说明那次恰好抽到了困难被试。跨度 7.6 个百分点，
又小于 135 个留出样本约 4.3 个百分点的标准误。**该批结果已撤下，不作为结论。**

有效版本需要：六结构 × 同一被试分组划分 × 至少 3 个种子。
现在有两条可执行路径：小数据用 `scripts/run_loso.py --arch <name>`（逐折留一），
全量语料用 `scripts/run_grouped_cv.py --arch <name>`（被试分组 K 折）。
脚本已就绪，六结构的结果**尚未产出**。

### 导联子集迁移（T1，LOSO 协议）：导联越少越差，且在 10 折上一致

协议同上一节：每折模型只在自己留出的被试上评测，10 折取均值。
同一套权重，只把部分通道置为无效（电极 ID 与坐标保留），模拟换一顶更稀疏的电极帽。
复现：`scripts/eval_montage_loso.py`，结果：[`reports/montage_transfer_loso.json`](reports/montage_transfer_loso.json)。

| 条件 | 保留导联 | BA（10 折均值） | Δ vs 全导联 | 变差的折数 |
|---|---|---|---|---|
| 全导联 | 64 | **0.6595** | — | — |
| 10-10 系统 | 32 | 0.5804 | −0.079 | 8/10 |
| 10-20 系统 | 19 | 0.5697 | −0.090 | 9/10 |
| 感觉运动区 | 8 | 0.5288 | −0.131 | **10/10** |
| BCIV-2b 布局 | 3 | 0.5200 | −0.140 | **10/10** |

随机丢弃作为对照（每折随机保留固定通道数，10 折均值）：32 导 0.5760、16 导 0.5616、8 导 0.5390。

**这是一条成立但方向与"鲁棒"相反的结论：**

- 通道数减少，性能单调下降；到 8 导 / 3 导时 **10 折全部下降**、平均损失 13–14 个百分点。
  所以**不能**声称"模型对稀疏导联鲁棒"，只能说"架构在数值上接受任意通道数"。
- 结构化移除与随机移除几乎等价（感觉运动区 8 导 0.5288 vs 随机 8 导 0.5390，差值落在折间标准差内），
  说明当前模型**没有真正利用电极拓扑**——这正是 `latent` 查询层与坐标编码应当补上的能力，但尚未体现为增益。
- 脑区整块移除：额区 −0.103（9/10 折变差）、枕区 −0.024、中央区 −0.012。
  额区影响最大与 MI 的生理先验不符，怀疑与参考电极/眼电残留有关，**需单独验证**，暂不作结论。

正交的结构性性质已验证：给 64 通道输入补 16 个标记为无效的通道，logits 差异严格为 0
（`tests/test_montage.py::test_padding_does_not_change_output`）——"任意通道数可进"在数值上成立，缺的是跨布局的性能保持。

**数据侧已经补齐（2026-11-21）**：BCI Competition IV 2a / 2b 已按同一协议接入，
产出 2 592（5 导）/ 6 520（3 导）个窗口，坐标解析率 1.000，并与 EEGMMIDB 合并出
全局通道词表（C3/C4/CZ 三库同 id）。三库样本可以在同一个 batch 里前向，坐标冲突为 0。
上表里的「BCIV-2b 布局」仍是**从 64 导里抽 3 个通道**的模拟子集，不是真实 2b 记录；
真实跨设备迁移的**训练/评测**尚未运行，因此这里不给出任何"迁移增益"的说法。

---

## 快速开始 / Quick Start

```bash
pip install -e ".[eeg]"
```

> **环境陷阱（踩过一次）**：本机上除了本仓库，还留着旧的
> `E:\世界机器人大赛\projects\脑电大模型\src` 代码树，而 `site-packages` 里的
> `__editable__.eeg_mime_foundation-0.1.0.pth` 曾经指向它。此时**不设 `PYTHONPATH`
> 的 `python -c` / `python script.py` 会静默导入那份旧代码**，报错内容还会指向不存在的
> 老 API（例如 `ModelConfig` 不认识 `graph`）。现已把该 `.pth` 重新指向本仓库，
> 迁移过环境后请先确认：
>
> ```bash
> python -c "import eeg_mime_foundation as m; print(m.__file__)"   # 必须落在本仓库 src 下
> ```

```bash
# 0. 下载全量公开语料（PhysioNet 开放许可，≈13 GB；断点续传 + SHA256 校验）
python scripts/download_corpora.py --dataset eegmmidb  --root D:/wyu-bci-data --jobs 16
python scripts/download_corpora.py --dataset sleep-edfx --root D:/wyu-bci-data --jobs 16

# 0.1 切窗并写入真实电极坐标（产出带坐标的 manifest）
python scripts/prepare_real_data.py ingest --dataset mi    --root D:/wyu-bci-data --scope corpus
python scripts/prepare_real_data.py ingest --dataset sleep --root D:/wyu-bci-data --scope corpus

# 0.15 跨设备语料（BCIC IV 2a/2b 在本地，不在 PhysioNet 链路上）
#      E session 官方只给 classlabel，必须配 --labels 才有标签
python scripts/prepare_real_data.py ingest --dataset bcic2a --scope corpus \
    --root D:/wyu-bci-data --raw "E:/情绪识别/data/BCICIV_2a/BCICIV_2a_gdf" \
    --labels "E:/情绪识别/data/BCICIV_2a/sets_2a_true_labels"
python scripts/prepare_real_data.py ingest --dataset bcic2b --scope corpus \
    --root D:/wyu-bci-data --raw "E:/情绪识别/data/BCICIV_2b/BCICIV_2b_gdf" \
    --labels "E:/情绪识别/data/BCICIV_2b/sets 2b true_labels"

# 0.16 标签交叉校验（T session 的事件码应与官方真值逐条相同）
python scripts/verify_bcic_labels.py \
    --raw "E:/情绪识别/data/BCICIV_2a/BCICIV_2a_gdf" \
    --labels "E:/情绪识别/data/BCICIV_2a/sets_2a_true_labels" \
    --output D:/wyu-bci-data/processed/corpus/bcic2a-corpus-v1/label_crosscheck.json

# 0.17 合并全局通道词表（不改任何 npz，只产出 local→global 重映射）
python scripts/build_shared_vocabulary.py \
    --manifest mi=D:/wyu-bci-data/processed/corpus/mi-corpus-v1/manifest.json \
    --manifest bcic2a=D:/wyu-bci-data/processed/corpus/bcic2a-corpus-v1/manifest.json \
    --manifest bcic2b=D:/wyu-bci-data/processed/corpus/bcic2b-corpus-v1/manifest.json \
    --output D:/wyu-bci-data/processed/corpus/shared-vocabulary-v1

# 0.18 跨库拼 batch 自检（词表、坐标、一次真实前向；不产出指标）
python scripts/check_cross_dataset.py \
    --vocabulary D:/wyu-bci-data/processed/corpus/shared-vocabulary-v1 \
    --manifest mi=<mi manifest> --manifest bcic2a=<2a manifest> \
    --output D:/wyu-bci-data/processed/corpus/shared-vocabulary-v1/cross_check.json

# 0.19 跨设备冻结迁移评测（不做微调，归一化统计取检查点里的 train-only 值）
#      按"电极名"把目标库映射到检查点的词表；映射率会写进 metrics.json
python scripts/run_cross_transfer.py \
    --checkpoint <mi best.pt> \
    --manifest D:/wyu-bci-data/processed/corpus/bcic2a-corpus-v1/manifest.json \
    --split validation --library bcic2a --output experiments/outputs/cross/bcic2a

# 0.2 被试分组交叉验证（大语料上的对照协议；--graph / --channel-dropout 做 A/B）
python scripts/run_grouped_cv.py --name cv-fusion --arch fusion \
    --manifest D:/wyu-bci-data/processed/corpus/mi-corpus-v1/manifest.json \
    --base-config configs/mi_corpus_supervised.json --folds 10 --epochs 40

# 结构消融（固定清单、固定 epoch、多种子；当前仅支持单次划分）
python scripts/run_arch_ablation.py \
    --manifest <manifest.json> --base-config <config.json> \
    --output reports/arch_ablation_mi.json --seeds 17 --epochs 40

# 跨被试留一（推荐协议，后续结论都以此为准）
python scripts/run_loso.py --name loso-fusion --arch fusion \
    --manifest <manifest.json> --base-config <config.json> --epochs 40

# 导联子集迁移（LOSO 协议）
python scripts/eval_montage_loso.py \
    --loso-root experiments/outputs/loso-fusion \
    --output reports/montage_transfer_loso.json --reps 10

# 导联子集迁移（单划分版本，仅用于快速冒烟）
python scripts/eval_montage_transfer.py \
    --checkpoint <best.pt> --manifest <manifest.json> \
    --output reports/montage_transfer.json --reps 20

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

- [`docs/能力与代码对应表.md`](docs/能力与代码对应表.md) — 每项能力的状态、代码位置与证据（哪些只是架构支持、哪些已有实验）
- [`docs/数据资产与缺口.md`](docs/数据资产与缺口.md) — 已有数据集、缺口、下载执行状态
- [`docs/全量语料与坐标协议.md`](docs/全量语料与坐标协议.md) — 109 被试语料、真实电极坐标、图偏置与导联丢弃的设计和待跑协议
- [`docs/architecture.md`](docs/architecture.md) — 编码器与数据流
- [`docs/5M_MI_ME模型改造方案.md`](docs/5M_MI_ME模型改造方案.md) — 参数预算与模块取舍
- [`docs/基础模型直接验证协议.md`](docs/基础模型直接验证协议.md) — 验证入口与门禁
- [`docs/训练前问题清单.md`](docs/训练前问题清单.md) — 启动正式预训练前的检查项
- [`docs/联合预训练与共享基座规范.md`](docs/联合预训练与共享基座规范.md) — MI + 睡眠共享基座

## 在项目群中的位置

训练与评测协议复用 [bci-bench](https://github.com/periodgod/bci-bench) 的数据契约、划分与指标定义；下游系统见 [bci-arm-robot](https://github.com/periodgod/bci-arm-robot) 与 [bci-visual-app](https://github.com/periodgod/bci-visual-app)。

---

<div align="center"><sub>Wuyi University · BCI Group　|　数据与权重不随仓库分发</sub></div>
