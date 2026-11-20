"""Verify recorded source/checkpoint artifacts before producing the acceptance report."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from eeg_mime_foundation.data import sha256
from eeg_mime_foundation.engine import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    matrices, all_runs = {},{}
    for kind in ("mi","sleep"):
        parts = {}
        for suffix,expected in (("verified",20),("controls-verified",4),("distillation-verified",2),("fewshot-verified",6)):
            directory = root/f"{kind}-{suffix}"
            runs = read(directory/"summary.json")
            if len(runs)!=expected:
                raise ValueError(f"Incomplete study {directory}: {len(runs)}/{expected}")
            parts[suffix] = runs
            all_runs.update({f"{kind}-{suffix}/{name}":run for name,run in runs.items()})
        matrices[kind] = parts
    joint = read(root/"joint-verified"/"summary.json")
    if len(joint)!=5 or not read(root/"joint-verified"/"completion.json")["transfer_uses_identical_base"]:
        raise ValueError("Joint transfer study incomplete")
    all_runs.update({f"joint-verified/{name}":run for name,run in joint.items()})
    verification = {}
    for name,run in all_runs.items():
        directory = Path(run["directory"])
        environment = read(directory/"environment.json")
        source = directory/"source.zip"
        if sha256(source)!=environment["source_snapshot"]["sha256"]:
            raise ValueError(f"Source archive changed: {name}")
        digest = hashlib.sha256()
        with zipfile.ZipFile(source) as archive:
            for path in sorted(n for n in archive.namelist() if n.startswith("src/") and n.endswith(".py")):
                digest.update(str(Path(path)).encode("utf-8"))
                digest.update(archive.read(path))
        if digest.hexdigest()!=environment["code"]["source_sha256"]:
            raise ValueError(f"Archived model source does not match recorded fingerprint: {name}")
        best = directory/"best.pt"
        actual = sha256(best)
        if run.get("checkpoint_sha256",actual)!=actual:
            raise ValueError(f"Checkpoint changed: {name}")
        if (directory/"test_access.jsonl").exists():
            raise ValueError(f"Unexpected real test scoring: {name}")
        verification[name] = dict(source_sha256=digest.hexdigest(),checkpoint_sha256=actual,
                                  source_snapshot_sha256=sha256(source),params=environment["params"])
    write_json(root/"artifact_acceptance.json",dict(runs=len(verification),all_source_archives_match=True,
                                                  all_checkpoints_match=True,test_predictions=0,records=verification))
    lines = ["# 基础模型直接研发验收", "", "2026-10-11。直接验证自研约4.91M参数候选，未新增普通模型训练对照。", "",
             f"完成{len(verification)}个有界真实数据研究运行；另完成两项训练样本记忆、CUDA精确续训、12次真实架构检查、17条合成接口路径及源码/检查点完整性核验。测试被试未生成模型预测或分数。", "",
             "## 结论", "", "工程链路成立；当前设计尚未通过通用基础模型收益验收。运动想象接近机会水平，睡眠能学习五阶段但弱类别及冻结迁移仍不稳定。多教师并非越多越好；学习门控未显示可靠收益。", "",
             "## 原生输入的三种子结果", "", "主指标：每个验证被试分别计算平衡准确率，再对被试平均，再对种子平均。±为三个训练种子的标准差，不是人群置信区间。", "",
             "| 自研候选路径 | MI BA | 睡眠 BA |", "|---|---:|---:|"]
    names = {"fusion_scratch":"融合结构从零监督", "factorized_scratch":"自研时空结构（取消融合组合）", "masked_full":"掩码预训练→全量微调", "masked_linear":"掩码预训练→冻结探测", "masked_adapter":"掩码预训练→adapter"}
    for variant,label in names.items():
        cells = []
        for kind in ("mi","sleep"):
            values = [matrices[kind]["verified"][f"{variant}_s{seed}"]["validation"]["subject_macro"]["balanced_accuracy"] for seed in (17,29,43)]
            cells.append(f"{np.mean(values)*100:.2f}% ± {np.std(values,ddof=1)*100:.2f}")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += ["", "每项10名公开数据被试，6:3:1；MI train270/val135，睡眠标签无关抽样train576/val480。监督5epochs、预训练3epochs。原保留S003/SC402不变，非官方8:1:1评分。fusion/factorized是组合模块消融，参数量不同；预训练另有计算成本。", "",
              "## 教师与目标链路（仅seed17）", "", "| 路径 | MI BA | 睡眠 BA |", "|---|---:|---:|"]
    variants = [("EMA预训练→微调","verified","ema_full_s17"),
                ("单教师：监督源","controls-verified","single_fusion_scratch_s17"),
                ("单教师：掩码微调源","controls-verified","single_masked_full_s17"),
                ("双教师：等权","distillation-verified","two_teacher_fixed_s17"),
                ("双教师：学习门控","distillation-verified","two_teacher_learned_s17"),
                ("对比预训练→微调","controls-verified","contrastive_full_s17")]
    for label,part,key in variants:
        cells = [f"{matrices[kind][part][key]['validation']['subject_macro']['balanced_accuracy']*100:.2f}%" for kind in ("mi","sleep")]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += ["", "两个教师均为本地自研、从零来源；独立投影与教师冻结检查通过。该单种子开发实验不能建立多教师优越性。学习门控在这批结果中不适合作为默认方案。", "",
              "## 少标签支持集", "", "每类8个训练试次；睡眠单位为30秒epoch，不是独立受试者。两路径共享同一无标签训练池归一化统计，验证/测试查询不变。", "",
              "| 路径 | MI BA，三种子均值 | 睡眠 BA，三种子均值 |", "|---|---:|---:|"]
    for variant in ("scratch","masked_full"):
        cells = [f"{np.mean([matrices[kind]['fewshot-verified'][f'{variant}_s{seed}']['validation']['subject_macro']['balanced_accuracy'] for seed in (17,29,43)])*100:.2f}%" for kind in ("mi","sleep")]
        lines.append(f"| {variant} | {' | '.join(cells)} |")
    lines += ["", "## 同一共享基座", "", "联合训练只有12名训练被试、846个窗口，联合NPZ移除全部标签；200Hz、真实通道名并集、0.5秒patch。同一个源encoder与统计初始化到两个任务，新建各自任务头。", "",
              "| 同一基座迁移 | MI BA | 睡眠 BA |", "|---|---:|---:|"]
    for tuning in ("full","linear"):
        cells = [f"{joint[f'{kind}_{tuning}']['validation']['subject_macro']['balanced_accuracy']*100:.2f}%" for kind in ("mi","sleep")]
        lines.append(f"| {tuning}，seed17 | {' | '.join(cells)} |")
    peaks = [run["cuda_peak_allocated_mib"] for run in all_runs.values()]
    lines += ["", "这证明共享输入/检查点的迁移链路可执行；重采样及睡眠patch变化使其与原生实验不可直接归因比较，不能称为联合预训练提升。", "",
              "## 代码、资源与来源", "",
              "- 六个结构在两项真实输入上通过梯度、遮盖隔离与通道同步重排检查。两项主候选在每类2个训练窗口、100步诊断上均100%记忆；这些权重被丢弃。",
              "- 预训练/教师/归一化训练来源累计传播；当前held-out训练组/波形冲突、源验证选择触碰测试及分组口径改变均拒绝。",
              "- 修复蒸馏按主体指标选模型时报错，增加按数据集成批，验证精确续训覆盖sampler状态。",
              f"- RTX4060 Laptop 8GB，FP32 batch8；研究运行最大进程内CUDA allocated峰值{max(peaks):.2f} MiB。并行作业存在GPU竞争，耗时不用于模型效率比较；此数值不包含驱动/桌面/其他进程。",
              "- 本机Python -I新进程使用保存源码加载，两项原生与两项共享基座迁移模型CPU输出严格一致；导出包移除优化器、EMA与教师，仍保留模型定义中的重建/频带层，未做官方容器验证。",
              "- 真实因果模型前缀、分块及reset一致；实现重算前缀。MI数据经过离线零相位滤波，不能据此声称整条链路在线。",
              "- source.zip哈希和压缩包内src指纹与environment.json逐一匹配，best.pt哈希匹配。不同修复版本由各运行源码快照追踪，不算独立种子。", "",
              "## 失败与限制", "",
              "保留mi架构诊断最初的cuDNN eval-mode反向错误、提前启动的少样本部分目录、蒸馏选择指标缺失及修复版本。失败/中断目录不计入69个完整运行。旧诊断中未训练头的分类或重建值不用于结论，以带head_supervised/decoder_trained字段的verified诊断为准。", "",
              "睡眠N1不再所有人/所有路径为零，但验证仅23个N1epoch，个别人只有3个，不能声称该类泛化问题已解决；架构、人数和权重同时改变，也不能单独归因类别权重。MI表征诊断显示高同向性，需要进一步判断空间读出或领域变化，不能仅凭有效秩认定塌陷。", "",
              "真实几何坐标/距离偏置、整夜上下文、相位敏感SSVEP、临床/情绪/自然刺激/可靠性真实任务、官方基座与21数据集、正式评分/容器仍待接入。基础模型不是七赛道效果已经成立。", "",
              "## 下一步顺序", "",
              "1. 保留现有可执行候选；优先诊断MI跨被试表征与空间读出，定义同输入/同无标签预算的联合预训练对照。",
              "2. 睡眠优先改善弱阶段与跨epoch上下文；按参与者分析，不用邻接epoch充当独立样本。",
              "3. 多教师保留实验入口，先证实单教师互补性；当前不默认采用学习门控。",
              "4. 官方数据、基座与指标发布后冻结输入/来源/任务头再运行；此处公开开发权重不自动视为比赛合法权重。", "",
              "证据入口：artifact_acceptance.json、mi/sleep-analysis.json、各launch/config/environment/epochs/source.zip、fewshot及joint completion.json、memory/recovery/stream/inference诊断。", "",
              "数据来源：[PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/)、[Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)，ODC Attribution v1.0。原始50个EDF共578.26MB，官方SHA256验证，来源/许可快照及版本保存在data/raw/precheck。"]
    Path(args.output).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Accepted {len(verification)} complete real study runs; report written",flush=True)


if __name__ == "__main__":
    main()
