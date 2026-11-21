"""把多个 manifest 的通道词表合并成全局空间，并输出每库的重映射表。

为什么必须有这一步：每个 manifest 的 `channel_vocabulary` 都是独立从 0 编号的。
EEGMMIDB 的 C3 是 5，BCIC IV 2a 的 C3 是 1。模型用 channel_id 去查电极 embedding，
直接混 batch 会让同一个电极拿到两行不同的参数——训练能跑完，指标也没报错，
但"跨设备迁移"测的其实是两个互不相干的传感器。

这里不改任何 npz：只产出
  shared_vocabulary.json   全局名字 → 全局 id（按名字排序，稳定可复现）
  remap-<name>.json        该库 local id → 全局 id
  summary.json             各库覆盖、共同通道、差集
下游在 __getitem__ 里做一次 id 映射即可。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_vocabulary(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = sorted((int(key), value) for key, value in manifest["channel_vocabulary"].items())
    if [index for index, _ in pairs] != list(range(len(pairs))):
        raise ValueError(f"{path} 的通道 id 不是从 0 连续编号")
    names = [name for _, name in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} 的通道名有重复")
    return names


def build(libraries):
    """libraries: {库名: [通道名, ...]} → 全局词表与每库重映射。"""
    universe = sorted({name for names in libraries.values() for name in names})
    ids = {name: index for index, name in enumerate(universe)}
    remaps = {label: {str(local): ids[name] for local, name in enumerate(names)}
              for label, names in libraries.items()}
    summary = dict(global_size=len(universe), libraries={}, shared=None, missing={})
    common = None
    for label, names in libraries.items():
        present = set(names)
        mapped = {name for name in present if name in ids}
        summary["libraries"][label] = dict(
            size=len(names), mappable=len(mapped),
            unmappable=sorted(present - mapped),
            global_ids=sorted(ids[name] for name in present))
        common = present if common is None else common & present
    summary["shared"] = sorted(common or [])
    for label, names in libraries.items():
        summary["missing"][label] = sorted(set().union(*[set(other) for other in libraries.values()])
                                          - set(names))
    return dict(frame="mne_head_cartesian; standard_1020 template", names=universe, ids=ids), remaps, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True,
                        help="库名=manifest.json，可重复，例如 mi=D:/.../mi-corpus-v1/manifest.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    libraries = {}
    for item in args.manifest:
        if "=" not in item:
            raise SystemExit(f"--manifest 需要 库名=路径 的形式：{item}")
        label, path = item.split("=", 1)
        libraries[label] = load_vocabulary(path)
    vocabulary, remaps, summary = build(libraries)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "shared_vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, remap in remaps.items():
        (output / f"remap-{label}.json").write_text(
            json.dumps(remap, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(dict(global_size=summary["global_size"],
                          shared=summary["shared"],
                          per_library={k: v["size"] for k, v in summary["libraries"].items()}),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
