> 历史方案（已停止作为执行入口）：旧 BSG 模型与固定输入配置已删除。当前代码、配方、验证和剪枝规则以 [赛前代码底座与剪枝树](赛前代码底座与剪枝树.md) 为准。本文保留初期研究设计，不代表当前实现或已验证效果。

# BSG 轻量 MI+ME 模型改造方案

本文档把项目第一版模型从 10M 通用 EEG encoder 收敛为 5M 以下的 MI+ME 领域基础模型。当前默认模型参考图片中的 “Lightweight Transformer for EEG Classification via Balanced Signed Graph Algorithm Unrolling” 思路，将普通注意力 Transformer 替换为 **BGL 图学习 + LPF 低通图滤波** 的算法展开结构。

## 1. 核心判断

只用 PhysioNet Motor Movement/Imagery 做预训练时，数据量约 43.6 小时任务 EEG。这个规模更适合：

```text
5M 以下 MI/ME 领域基础模型
```

不适合作为唯一数据源训练更大的通用 EEG 大模型。

## 2. 架构目标

模型不再追求“泛化所有 EEG 范式”，而是优先契合：

- 真实运动 ME。
- 运动想象 MI。
- 左手、右手、双手、脚。
- 极少样本 8 分类快速适配。
- 感觉运动区 C3/Cz/C4 附近的 mu/beta 节律。

## 3. 推荐架构

名称：

```text
BalancedSignedGraphMIME
```

输入：

```text
[batch, 64 channels, 4 seconds * 160 Hz]
= [B, 64, 640]
```

patch：

```text
0.5 秒 patch
80 samples / patch
8 time patches
64 * 8 = 512 channel-time tokens
```

主干：

```text
d_model = 192
depth = 6
graph_topk = 12
propagation_steps = 3
block = BGL + balanced signed LPF + temporal LPF + FFN
```

参数量：

```text
总参数 1,393,642
低于 5M 预算，适合 PhysioNet 单数据集预训练
```

## 4. 为什么不用普通 Transformer

普通 Transformer 会把 512 个 channel-time token 当作无结构序列。这样有三个问题：

1. 它不知道 C3/C4/Cz 对 MI/ME 更重要。
2. 它不知道时间动态和通道空间关系是两种不同结构。
3. 它容易在小数据上学习设备或被试噪声。

因此改为平衡符号图算法展开：

```text
BGL:
  根据每个时间片内的通道 token 学习通道图。

Balanced signed graph:
  学习每个通道 token 的极性，用符号变换把 signed graph 转为 positive graph。

LPF:
  在 positive graph 上做低通图滤波，再映射回原信号空间。

Temporal LPF:
  用轻量 depthwise temporal filter 保留 ERD/ERS 时间连续性。
```

这比全局 attention 更符合小数据 MI+ME，也更省参数和算力。

## 5. MI+ME 先验如何注入

### 5.1 频带感知 patch embedding

每个 patch 不只输入 raw waveform，还额外计算：

```text
mu: 8-12 Hz
low_beta: 13-20 Hz
high_beta: 20-30 Hz
```

模型将 raw patch embedding 与 bandpower embedding 相加。

### 5.2 运动区通道偏置

训练时可以对这些通道相关 patch 提高 mask 概率或 loss 权重：

```text
C3, Cz, C4, FC3, FCz, FC4, CP3, CPz, CP4
```

这不是硬编码答案，而是让模型更关注 MI/ME 的主要生理区域。

### 5.3 层级分类头

不要一开始只做 flat 8-class。推荐三个头：

```text
state head: MI vs ME
body head: left_hand / right_hand / both_hands / feet
joint head: 8-class
```

这样可以定位之前接近随机的问题来自哪里：

- 如果 state 也随机，说明 MI/ME 区分失败。
- 如果 state 可行但 body 随机，说明动作空间模式失败。
- 如果两个子任务都可行但 8 类不稳，说明组合分类和小样本适配要优化。

## 6. 预训练任务

主任务：

```text
masked channel-time reconstruction
```

损失：

```text
raw waveform MSE
+ spectral/log-PSD MSE
+ mu/beta bandpower MSE
```

建议权重：

```text
raw: 1.0
spectral: 0.3
bandpower: 0.5
```

## 7. 实验顺序

必须按这个顺序做：

1. FBCSP-LDA / Riemannian / EEGNet baseline。
2. 1M-2M sanity model。
3. BSG 轻量主模型。
4. 5M/10M 上限对照，仅在主模型有效后再做。

如果 BSG 轻量主模型不能超过传统 baseline，不应继续堆参数。

## 8. 当前实现入口

- 配置：`configs/model_5m_mime.yaml`
- 模型代码：`src/eeg_mime_foundation/model.py`
