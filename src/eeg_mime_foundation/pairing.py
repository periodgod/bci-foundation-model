"""PSG 与睡眠分期标注的配对（纯标准库，供下载器与 ingest 共用）。

Sleep-EDF 的标注文件名不能靠替换 PSG 文件名推导：建库时版本位被逐个改过，
`SC4001E0` 配 `SC4001EC`，`SC4011E0` 配 `SC4011EH`，`ST7011J0` 配 `ST7011JO`。
早期实现按 `C0123456` 逐个试字母，命中不了就静默跳过——结果是整批睡眠数据
在没有报错的情况下少了一大半。这里改为按"子库目录 + 6 位编号"建索引，
缺一个或撞键都抛错。
"""

from pathlib import Path


def _key(name):
    path = Path(name)
    return path.parent.name, path.stem[:6]


def pair_hypnograms(psg_names, hypnogram_names):
    """返回 [(psg, hypnogram)]；配对失败抛 ValueError，不返回残缺清单。"""
    index = {}
    for name in hypnogram_names:
        index.setdefault(_key(name), []).append(name)

    pairs, missing, ambiguous = [], [], []
    for psg in psg_names:
        found = index.get(_key(psg), [])
        if len(found) > 1:
            ambiguous.append((psg, sorted(found)))
        elif not found:
            missing.append(psg)
        else:
            pairs.append((psg, found[0]))
    if ambiguous:
        raise ValueError(f"标注文件前缀不唯一，无法配对: {ambiguous[:3]}")
    if missing:
        raise ValueError(f"以下 PSG 找不到分期标注: {missing[:3]}（共 {len(missing)} 条）")
    return pairs


def scan_sleep_pairs(directory):
    """扫描 sleep-cassette 与 sleep-telemetry 两个子库，返回相对路径配对。"""
    directory = Path(directory)
    psg, hypnograms = [], []
    for sub in ("sleep-cassette", "sleep-telemetry"):
        psg += [f"{sub}/{p.name}" for p in sorted((directory / sub).glob("*-PSG.edf"))]
        hypnograms += [f"{sub}/{p.name}" for p in sorted((directory / sub).glob("*-Hypnogram.edf"))]
    return pair_hypnograms(psg, hypnograms)
