# 数据账本与真实数据预检

实际账本沿用 JSON manifest 与逐文件下载记录，不另建独立 CSV 状态来源。

| 位置 | 内容 |
|---|---|
| raw/precheck/eegmmidb/1.0.0 | S001–S010，各 R04/R08/R12，共30 EDF；原三人下载记录与 development 记录分别保存 |
| raw/precheck/sleep-edfx/1.0.0 | SC400–SC409 第一夜 PSG 与配对人工 hypnogram，共20 EDF；原记录与扩展记录分别保存 |
| processed/precheck/mi-v1 | 64 通道 × 560 样本、160 Hz、左右手标签 NPZ 与 manifest/protocol/intake_audit |
| processed/precheck/sleep-v1 | Fpz-Cz/Pz-Oz 双极导联、100 Hz、30 秒 epoch、五阶段标签及对应账本 |
| processed/precheck/mi-development-v1 | 10 人、450 窗口；train270/validation135/test45；原 S003 保留 |
| processed/precheck/sleep-development-v1 | 10 人、9913 epochs；train5519/validation3369/test1025；原 SC402 保留 |

原始文件保持不变。下载记录含官方 SHA-256 与实际传输地址；只有校验完整的文件才去掉 `.part` 后缀。公开数据仅获准用于本地预检，比赛训练许可仍须按正式赛规核验。

MNE EDF 输出伏特，不重复乘 1e-6。坐标未知时明确 coord_valid=false；用真实通道/双极导联名称映射词表，不编造电极位置。样本记录保留原始文件、被试、会话、试次与起止采样点；被试在切窗前分配，归一化和 CSP/特征缩放仅拟合 train。

原三人诊断采用1:1:1，扩展十人开发采用6:3:1；都不是官方8:1:1完整研究。扩展原始文件均通过官方SHA256校验。原衍生版本不覆盖，保留测试被试不改变，测试集合仅完整性审计，不生成模型预测或参与选择。

基础模型直接实验另在 experiments/outputs/foundation-direct-20261011 各 study manifest 记录有界抽样：MI全训练/验证，睡眠训练最多96 epochs/人、验证160 epochs/人，标签无关固定随机选择；共576/480训练/验证epochs。不能把这部分结果当全9913epochs训练结果。

```powershell
py -3.12 -X utf8 scripts/prepare_real_data.py download --dataset all
py -3.12 -X utf8 scripts/prepare_real_data.py ingest --dataset all
py -3.12 -X utf8 scripts/prepare_real_data.py download --dataset all --scope development
py -3.12 -X utf8 scripts/prepare_real_data.py ingest --dataset all --scope development
py -3.12 -X utf8 scripts/validate_foundation.py --dataset mi --manifest data/processed/precheck/mi-development-v1/manifest.json --output experiments/outputs/new-foundation-mi
```

已完成的衍生目录不得覆盖；已有有效原始文件可通过哈希校验后复用。训练输出须使用新目录。安装项声明在 pyproject.toml 的 eeg extra，本机已存在相应依赖，本次未安装新包。

实际验收状态见 [项目阶段记录](../docs/赛前准备记录模板.md)。
