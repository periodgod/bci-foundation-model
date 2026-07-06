# 10M EEG Foundation Encoder 架构

## 1. 为什么从这个架构开始

脑电数据的难点不是单纯序列很长，而是：

- 不同数据集通道数不同。
- 不同设备采样率不同。
- 不同被试差异很大。
- 有标签数据少，无标签连续数据相对更容易积累。

因此第一版应优先选择易训练、易迁移、易评估的自监督 encoder。

推荐方案：**channel-time patch Transformer**。

## 2. 输入表示

统一预处理：

- 重采样到 `200 Hz`。
- 默认带通 `0.5-45 Hz`，下游 MI 可另用 `8-30 Hz`。
- 50 Hz notch。
- 平均参考或数据集原始参考需记录。
- 每段裁成 `4s`。

张量形状：

```text
x: [batch, channels, time_samples]
默认: [B, 32, 800]
```

patch 化：

```text
patch_seconds = 1.0
patch_samples = 200
num_time_patches = 4
num_tokens = channels * num_time_patches
```

每个 token 对应：

```text
某一个通道的一段 1 秒 EEG
```

## 3. Patch Embedding

第一版用轻量 1D Conv/MLP：

```text
[B, C, T] -> [B, C, P, patch_samples] -> [B, C*P, d_model]
```

每个 patch 加三类 embedding：

- `time_pos_embedding`
- `channel_embedding`
- 可选 `dataset_embedding`

通道编码第一版用 learnable channel ID。后续可加入 10-20 电极坐标，用几何位置编码解决跨 montage 问题。

## 4. Encoder

默认参数：

```yaml
d_model: 256
depth: 8
num_heads: 8
mlp_ratio: 4
dropout: 0.1
attn_dropout: 0.1
norm: layernorm
activation: gelu
```

参数估算：

```text
Transformer encoder: 约 6.3M
patch/channel/time embeddings: 约 0.2M-0.5M
pretraining decoder/head: 约 2M-3M
总训练参数: 约 9M-11M
部署 encoder: 约 7M
```

## 5. 预训练任务

主任务：masked channel-time reconstruction。

步骤：

1. 随机 mask 50%-70% 的 channel-time patch。
2. encoder 只看未 mask token。
3. 小 decoder 重建被 mask 的 EEG patch。
4. 同时约束时域和频域。

损失：

```text
loss = mse(raw_patch, recon_patch) + 0.2 * mse(log_psd(raw_patch), log_psd(recon_patch))
```

第二阶段可加入：

- subject contrastive loss
- dataset adversarial/domain loss
- masked bandpower prediction
- supervised multitask heads

## 6. 微调与评估

必须同时评估：

- within-subject
- cross-subject
- leave-one-subject-out

下游任务：

- MI：body part / imagery vs execution
- SSVEP：频率识别
- emotion EEG：valence/arousal/class label
- ERP/P300：事件检测
- robot control：实时少样本适配

与这些 baseline 对比：

- FBCSP-LDA
- Riemannian tangent space
- EEGNet
- MiniRocket
- 小型 CNN/TCN

## 7. 不推荐第一版做什么

- 不先做 50M+ 参数模型。
- 不先做复杂 VQ tokenizer。
- 不用 trial shuffle 跨被试混合验证。
- 不用重叠窗口制造虚高样本量，若使用必须单独报告。
- 不把所有数据强行插值到同一通道布局后不记录来源。

