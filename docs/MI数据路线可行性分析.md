# MI 数据路线可行性分析

本文分析是否可以不用 TUEG，改用 BCI Competition IV 2a 和 PhysioNet EEG Motor Movement/Imagery 作为脑电大模型第一阶段数据。

## 结论

可以不用 TUEG 作为第一阶段数据，但要重新定义项目目标：

```text
不是训练通用 EEG foundation model，
而是训练面向运动想象/运动执行的 MI/ME domain foundation model。
```

推荐路线：

```text
PhysioNet Motor Movement/Imagery
  -> MI/ME 领域自监督预训练主数据

BCI Competition IV 2a
  -> 标准 MI benchmark 与领域适配数据

我们自采 8 分类 MI+ME
  -> 最终极少样本快速微调目标任务
```

## 数据规模

| 数据集 | 被试 | 通道 | 采样率 | 任务 | 粗略有效规模 | 角色 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| PhysioNet EEG Motor Movement/Imagery | 109 | 64 | 160Hz | ME + MI + baseline | 约 43.6 小时 ME/MI 任务段，约 47.2 小时含 baseline | 预训练主数据 |
| BCI Competition IV 2a | 9 | 22 EEG + 3 EOG | 250Hz | 4 类 MI | 约 5.76 小时 4s MI 标签段 | benchmark / 领域适配 |

说明：

- PhysioNet 每名被试 14 个 run：2 个 1 分钟 baseline，12 个 2 分钟任务 run。
- PhysioNet 的 12 个任务 run 包含真实运动和运动想象，因此比纯 TUEG 更贴近 MI+ME。
- BCI IV 2a 每名被试 2 个 session，每个 session 288 trial，4 类 MI：左手、右手、双脚、舌头。

## 对我们 8 分类 MI+ME 的贴合度

我们的目标如果是：

```text
状态：MI / ME
动作：左手 / 右手 / 双手 / 脚
组合：2 x 4 = 8 类
```

两个公开数据集的作用不同：

| 数据集 | 与 8 分类 MI+ME 的关系 |
| --- | --- |
| PhysioNet | 最贴近。包含真实运动和运动想象，也包含左/右 fist、双 fist、双 feet。 |
| BCI IV 2a | 只含 MI，不含 ME；有左手、右手、双脚、舌头，不含双手。适合训练 MI 表征和做标准 benchmark。 |

因此，PhysioNet 是第一优先，BCI IV 2a 是第二优先。

## 是否足够训练 10M 模型

谨慎结论：

```text
足够做 10M 模型的 MI/ME 领域预训练原型；
不足以单独支撑真正通用的 EEG 大模型。
```

原因：

- PhysioNet + BCI IV 2a 的有效任务时长大约 50 小时量级。
- 对 10M 参数模型来说，这个规模偏小，容易过拟合数据集风格、设备和范式。
- 但因为它们都属于运动相关 EEG，任务一致性高，适合做 domain-specific pretraining。

建议把第一阶段目标写成：

```text
训练一个面向 MI/ME 的脑电基础编码器，
验证其在极少样本 8 分类 MI+ME 任务上的快速适配能力。
```

不要写成：

```text
训练通用 EEG-LLM 大模型。
```

## 推荐训练路线

### 阶段 1：PhysioNet 自监督预训练

- 输入：连续 EEG 或按事件切出的 4s/6s 片段。
- 任务：masked channel-time reconstruction。
- 可保留 ME、MI、baseline，但训练时记录 run type。
- 重点学习运动相关脑电的通用时间、频谱、通道结构。

### 阶段 2：BCI IV 2a 监督/自监督领域适配

- 使用 4 类 MI 标签。
- 做 linear probe、full fine-tuning、few-shot fine-tuning。
- 对比 FBCSP-LDA、Riemannian、EEGNet。

### 阶段 3：自采 8 分类 MI+ME 极少样本微调

推荐不要直接 flat 8-class，而是层级建模：

```text
Head 1: MI vs ME
Head 2: 左手 / 右手 / 双手 / 脚
Final: 状态 x 动作 = 8 类
```

## 模型规模建议

如果只用 PhysioNet + BCI IV 2a：

| 模型 | 建议 |
| --- | --- |
| 1M-3M | 最稳，适合快速验证 |
| 5M-10M | 可行，但必须强正则、强数据增强、严格验证 |
| 10M+ | 不建议作为第一版，除非加入更多 MI/ME 数据 |

当前项目仍可保留 10M 目标，但应加入小模型对照：

```text
3M baseline encoder
10M main encoder
FBCSP/Riemannian/EEGNet traditional baselines
```

## 关键风险

1. PhysioNet 与我们自采设备、通道、任务节奏可能不同。
2. PhysioNet 的动作标签结构与我们的 8 分类并非完全一致。
3. BCI IV 2a 没有 ME，只能帮助 MI。
4. 公开数据总量不大，不能用窗口重叠虚增独立样本。
5. 如果我们自采数据质量低或事件对齐不准，预训练也救不了最终随机结果。

## 当前建议

第一阶段先不使用 TUEG。

更聚焦地写成：

```text
基于公开运动想象/运动执行 EEG 数据的 MI/ME 脑电基础模型，
通过自监督预训练、教师蒸馏和参数高效微调，
实现自采 8 分类 MI+ME 任务的极少样本快速适配。
```

