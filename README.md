# 脑电与时序基础模型：赛前代码准备

目前准备的是可运行、可替换、可验证的研究底座。用户提供的赛事公告已明确七赛道、2–3 人组队、复赛官方基座微调和决赛从零预训练；具体数据、指标、许可和部署限制仍待正式规则确认。运动想象与睡眠已各扩展到10名被试，并直接验证约4.91M参数的自研融合候选；真实数据与代码通过不等于基础模型泛化优势成立。本项目候选不是 CBraMod、EEGPT、LUNA 等论文的完整复刻。

**执行规范：[赛前代码底座与剪枝树](docs/赛前代码底座与剪枝树.md)**。

按用户最新要求，当前以[基础模型直接验证协议](docs/基础模型直接验证协议.md)为研发入口，暂停新增普通模型对照。[七赛道接入边界](docs/七赛道接入与研发验收边界.md)逐项记录已实现能力与待接入内容。

最新[直接研发验收](reports/基础模型直接研发验收.md)：69个完整真实数据研究运行已核验源码与检查点；三种子MI直接训练BA51.72%、掩码全量微调51.37%，睡眠对应67.17%/69.18%。同一联合预训练基座完成两个任务的full/linear迁移。代码可运行，通用基础模型收益仍未成立。

| 模块 | 已实现 |
|---|---|
| 数据 | NPZ/JSON 账本、真实通道身份、可变长度与通道、分组审计、哈希、训练统计 |
| 编码器 | 波形频谱融合、时空分解、轻量卷积、潜在通道、长序列分层、因果 GRU |
| 输出 | 窗口分类/回归、patch 序列分类/回归、重建与频带目标 |
| 训练 | 监督、掩码重建/频谱/频带、EMA、双视图对比、多教师特征/同标签输出蒸馏；训练标签频率权重 |
| 适配 | 从头训练、编码器初始化、全量微调、线性探测、残差 adapter、试次级少样本 |
| 验证 | 预测与元数据、主体汇总、测试访问账本、因果参考推理、批次资源测量 |
| 复现 | 配置/源码/数据哈希、环境、日志、检查点、epoch 边界恢复、上游训练与模型选择来源链 |
| 剪枝 | 规则/资源约束、配对验证收益与主体 bootstrap、关键任务保护；缺证据保持待验证 |

## Windows 启动

本机已验证 Python 3.12、PyTorch 2.6.0+cu124、NumPy 2.5.3。使用现有依赖及标准库 unittest。

```powershell
py -3.12 -m pip install -e . --no-deps --no-build-isolation
py -3.12 -X utf8 -m unittest discover -s tests -v
py -3.12 -X utf8 -m eeg_mime_foundation.cli smoke --output experiments/outputs/my-smoke
```

新运行使用新目录。smoke 只用小模型和合成数据检查接口，不据其分数选择比赛方案。

直接验证真实尺寸的自研候选：

```powershell
py -3.12 -X utf8 scripts/validate_foundation.py --dataset mi --manifest data/processed/precheck/mi-development-v1/manifest.json --output experiments/outputs/my-foundation-mi
py -3.12 -X utf8 scripts/validate_foundation.py --dataset sleep --manifest data/processed/precheck/sleep-development-v1/manifest.json --output experiments/outputs/my-foundation-sleep
```

```powershell
py -3.12 -X utf8 -m eeg_mime_foundation.cli synthetic --output data/processed/diagnostic
py -3.12 -X utf8 -m eeg_mime_foundation.cli train --config configs/fusion_masked.json --manifest data/processed/diagnostic/manifest.json --output experiments/outputs/masked
py -3.12 -X utf8 -m eeg_mime_foundation.cli train --config configs/fusion_supervised.json --manifest data/processed/diagnostic/manifest.json --initialize experiments/outputs/masked/best.pt --output experiments/outputs/finetuned
py -3.12 -X utf8 -m eeg_mime_foundation.cli evaluate --checkpoint experiments/outputs/finetuned/best.pt --manifest data/processed/diagnostic/manifest.json --split validation --output experiments/outputs/finetuned-validation
py -3.12 -X utf8 -m eeg_mime_foundation.cli decide --evidence configs/decision_pending.json --output experiments/outputs/decision.json
```

## 模型讲解动画

后续动画统一使用 Remotion；可编辑工程、中文旁白和导出说明见 [animations/README.md](animations/README.md)。

```powershell
cd animations
npm.cmd ci
npm.cmd run narrate
npm.cmd run studio
# 导出 1080p 中文讲解视频
npm.cmd run render
```

## 文档

- [代码底座与剪枝树](docs/赛前代码底座与剪枝树.md)：接口、配方、决策条件、开赛修改顺序。
- [工程验收记录](reports/赛前代码工程验收.md)：测试结果与限制。
- [真实数据闭环验收](reports/运动想象与睡眠真实数据闭环验收.md)：下载/原始标签、领域基线、训练验证、资源、睡眠 N1 缺口与测试访问范围。
- [全局技能一致性检查](reports/全局技能一致性检查.md)：49 个相关入口检查、8 个技能修订与项目状态对齐。
- [赛事公告与真实数据验证计划](docs/赛事公告与赛前真实数据验证计划.md)：两阶段权重约束、七赛道缺口、最小数据清单与初赛前安排。
- [时序与 EEG 融合预案](docs/时序与EEG融合架构及赛前总预案.md)：S0–S7。
- [文献地图与模块替换](docs/赛前文献地图与模块替换计划.md)。
- [多教师与任务迁移](docs/多教师蒸馏与任务迁移方案.md)。
- [跨领域时序参考](docs/跨领域时序参考与架构预案.md)。
- [赛前流程规范](docs/赛前准备流程规范.md)、[记录模板](docs/赛前准备记录模板.md)、[技能更新报告](docs/技能更新与验证报告.md)。
- [OmniEEG 初期盘点](docs/OmniEEG基础模型全流程设计.md)：代码状态以本 README 为准。

旧 BSG 原型、固定输入 YAML 和参数统计脚本已删除，没有兼容别名。之前的 BSG/MI 设计保留为历史证据，不能作为当前运行规范。真实数据预检使用 [数据账本](data/README.md) 中的限定导入流程与领域基线。外部官方模型原生输入适配、优化流式缓存、未知任务识别和官方提交封装仍需实际任务确定后实施。
