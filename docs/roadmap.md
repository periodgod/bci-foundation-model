# 第一阶段路线

## Step 1：数据账本

先统计每个 EEG 数据源：

```text
dataset_name
task_type
subjects
sessions
channels
sampling_rate
recording_hours
license
labels
montage
quality_notes
```

优先目标：

- 先凑到 200 小时可跑 smoke pretraining。
- 500 小时进入正式 10M 预训练。
- 1000 小时作为第一阶段理想目标。

## Step 2：统一数据规范

统一为：

```text
float32 EEG
单位: microvolt 或 volt，必须记录
shape: [channels, time]
metadata: subject/session/task/dataset/sfreq/channel_names
```

预处理约束：

- 重采样前先低通抗混叠。
- notch 去 50Hz 工频。
- 不盲目 ICA。
- 不跨 epoch 做 baseline。
- train/val/test 按 subject 或 session 切分。

## Step 3：先做 baseline

在训练大模型前，必须得到基线：

- FBCSP-LDA
- Riemannian tangent space
- EEGNet

这样后面才能证明 10M 模型确实带来收益。

## Step 4：10M MAE 预训练

最小可行训练：

- 输入 4s。
- 32 通道。
- 200Hz。
- 1s patch。
- mask 60%。
- batch 尽可能大。
- 保存每 5%-10% 训练进度 checkpoint。

## Step 5：下游验证

优先用我们已有任务：

1. MI/ME：运动想象与执行。
2. SSVEP：频率识别。
3. 情绪 EEG/fNIRS 项目中的 EEG 部分。

评估方式：

- frozen linear probe
- full fine-tuning
- few-shot subject adaptation

