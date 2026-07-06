# PhysioNet 单数据集参数规模分析

本文分析：如果第一阶段只使用 PhysioNet EEG Motor Movement/Imagery Dataset 做预训练，模型参数量应该多大，以及原计划 10M 是否合理。

## 数据集规模

PhysioNet EEG Motor Movement/Imagery Dataset 的官方信息：

- 109 名被试。
- 64 通道 EEG。
- 160 Hz 采样率。
- 每名被试 14 个 run。
- 前 2 个 run 是 1 分钟 baseline：睁眼、闭眼。
- 后 12 个 run 是 2 分钟任务：真实运动 ME 与运动想象 MI。
- 文件格式为 EDF+，包含 annotation channel。

换算：

```text
任务数据:
109 subjects * 12 runs * 2 min = 2616 min = 43.6 h

含 baseline:
109 subjects * (12*2 + 2*1) min = 2834 min = 47.23 h
```

如果使用 4 秒非重叠窗口：

```text
43.6 h * 3600 / 4 = 39240 windows
```

如果采用 1 秒 channel-time patch：

```text
39240 windows * 64 channels * 4 time patches
= 10045440 channel-time tokens
约 10.0M tokens
```

如果看原始标量采样点：

```text
43.6 h * 3600 * 64 channels * 160 Hz
= 1.607B scalar samples
```

但注意：原始采样点高度相关，不能直接等同于语言模型里的独立 token。

## 与模型参数量的关系

只用 PhysioNet 时，数据规模属于：

```text
中等被试数
小到中等总时长
高任务相关性
单数据集、单设备/单范式偏置明显
```

它非常适合训练 MI/ME 领域模型原型，但不足以支撑真正通用 EEG 大模型。

## 推荐参数规模

| 模型规模 | 是否推荐 | 说明 |
| --- | --- | --- |
| 1M-2M | 推荐作为 sanity check | 快速验证数据、loader、mask 任务、下游微调是否有效 |
| 3M-5M | 最推荐作为主模型 | 与 43.6 小时任务数据更匹配，过拟合风险可控 |
| 7M-10M | 可作为上限实验 | 可以做，但必须强正则、强增强、严格跨被试验证 |
| >10M | 不推荐 | 单一 PhysioNet 数据很难支撑，容易记住数据集和被试特征 |

## 对原 10M 方案的判断

结论：

```text
10M 不是完全不合理，但如果只用 PhysioNet，它偏大。
```

更准确的定位是：

```text
3M-5M：第一阶段主力模型
10M：上限对照模型 / scaling ablation
```

如果 3M-5M 模型都不能在自采 8 分类 MI+ME 上超过 FBCSP、Riemannian、EEGNet，那么 10M 很可能不会自然解决问题。

## 推荐实验矩阵

第一阶段不要只跑一个 10M 模型，建议跑小网格：

| 实验 | 参数量 | 目的 |
| --- | ---: | --- |
| E0 | 传统 baseline | FBCSP、Riemannian、EEGNet |
| E1 | 1M-2M | 检查预训练任务是否有效 |
| E2 | 3M-5M | 主模型 |
| E3 | 7M-10M | 验证扩大参数是否带来收益 |

判断标准：

- PhysioNet held-out subjects 上 reconstruction loss 是否下降。
- PhysioNet ME/MI 线性 probe 是否优于随机和简单 CNN。
- 自采 8 分类 MI+ME 少样本微调是否优于传统 baseline。
- 跨被试和跨 session 结果是否稳定。

## 对自采 8 分类 MI+ME 的建议

自采任务建议不要直接 flat 8-class 起步。

优先使用层级结构：

```text
Head 1: MI vs ME
Head 2: 左手 / 右手 / 双手 / 脚
Final: 2 x 4 = 8 类
```

原因：

- PhysioNet 本身天然包含 ME/MI 状态和动作类别。
- 层级任务比直接 8 分类更适合极少样本。
- 如果之前结果接近随机，可以先定位是状态分类失败，还是动作分类失败。

## 最终建议

如果只用 PhysioNet 预训练：

```text
项目目标：MI/ME 领域基础模型
主模型规模：3M-5M
10M 定位：可选上限实验
正式训练前必须先跑 baseline 和小模型
```

保留 10M 可以，但不建议把 10M 作为唯一主线。

