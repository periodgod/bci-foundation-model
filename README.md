# 脑电大模型

目标：建设一个面向脑机接口和脑电智能分析的 EEG foundation model 项目。第一阶段收敛为 **5M 级 MI+ME 脑电基础模型**，优先服务运动想象、运动执行和自采 8 分类 MI+ME 极少样本快速适配任务。

本项目原则：**训练之前，先把数据、任务、架构、评估、伦理、算力和文档问题逐项解决清楚。**

## 当前架构决策

第一版采用 **MI+ME factorized channel-time Transformer encoder**：

- 输入：多通道 EEG 连续片段。
- tokenizer：每个通道按 0.5 秒切成 patch，融合 raw waveform 与 mu/beta bandpower。
- 位置编码：时间位置编码 + 通道编码。
- 主干：6 层分解式 Transformer，先做通道内 temporal attention，再做同一时刻的 spatial attention。
- 预训练：masked channel-time reconstruction，加入频谱和 bandpower 约束。
- 微调：MI/ME 状态头、身体部位头、8 分类联合头。

不在第一版采用大型 VQ tokenizer 或生成式 decoder，先把数据、预训练、评估闭环跑通。

## 推荐第一阶段规格

| 项目 | 建议 |
| --- | --- |
| 参数量 | 主模型 4,899,870 参数，10M 仅作为上限对照 |
| 输入长度 | 4 秒 |
| 采样率 | 160 Hz，优先贴合 PhysioNet 原生采样率 |
| 通道数 | 64，支持 channel mask |
| patch 长度 | 0.5 秒 |
| token 数 | 64 通道 x 8 时间片 = 512 tokens |
| 预训练数据 | PhysioNet Motor Movement/Imagery |
| mask 比例 | 默认 55% channel-time patches |
| 损失 | raw MSE + spectral loss + bandpower loss |

## 文件

- `docs/基础说明.md`：项目基础说明，先作为全项目共识文档。
- `docs/训练前问题清单.md`：训练前必须解决的问题和决策状态。
- `docs/5M_MI_ME模型改造方案.md`：5M MI+ME 架构改造说明。
- `docs/完整实验方案.md`：从数据到账本、baseline、预训练、微调、画图和报告的完整流程。
- `docs/architecture.md`：架构细节。
- `docs/roadmap.md`：第一阶段路线。
- `configs/model_5m_mime.yaml`：5M MI+ME 默认模型配置。
- `configs/model_10m.yaml`：默认模型配置。
- `src/eeg_mime_foundation/model.py`：5M MI+ME PyTorch 模型骨架。
- `data/README.md`：数据账本字段。
- `experiments/README.md`：每次实验目录格式。
- `results/README.md`：结果表保存规范。
- `figures/README.md`：图表保存规范。
- `reports/README.md`：报告和模型卡保存规范。

## 当前阶段

现在处于 **5M MI+ME 模型改造与预训练准备阶段**。

## 快速检查

```powershell
pip install -e .
python scripts/count_5m_model_params.py
```

期望输出：

```text
total_parameters=4,899,870
trainable_parameters=4,899,870
```

进入正式训练前，必须完成：

1. 数据账本。
2. 数据许可与隐私确认。
3. 预处理规范。
4. 模型输入规范。
5. 训练目标与损失设计。
6. 下游评估协议。
7. baseline 结果。
8. 算力与训练预算。
9. 实验记录与复现规范。
10. 风险清单和终止条件。

当前执行顺序以 `docs/完整实验方案.md` 为准。没有完成数据账本、传统 baseline 和 1M-2M sanity model 前，不启动 5M 主模型正式预训练。
