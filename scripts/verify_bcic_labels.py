"""交叉校验 BCIC IV 的事件码与官方真值。

T session 同时带 769/770（事件码）和 sets_*_true_labels 里的 classlabel。
两者本该逐条相同：如果有偏差，说明标签来源、+768 映射或顺序对齐有一处错了，
而这种错误在训练里只表现为"准确率偏低"，不会报错。

E session 没有事件码，无法自校验——它们的正确性依赖同一套 +768 约定，
所以这项检查是在为 E session 的正确性做背书，而不是只查 T。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation.realdata import bcic_truth_labels  # noqa: E402


def session_codes(path):
    """取全部四个类别码（769–772）与真值逐条比对。

    只取手部两类会让序列长度对不上（144 vs 288），进而误报不一致——
    这里要验证的是"整条序列没有位移"，所以必须比全量。
    """
    import mne

    raw = mne.io.read_raw_gdf(str(path), preload=False, verbose="ERROR")
    return [int(str(value).strip()) - 768 for value in raw.annotations.description
            if str(value).strip() in {"769", "770", "771", "772"}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="gdf 目录")
    parser.add_argument("--labels", required=True, help="官方真值 .mat 目录")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw_directory, label_directory = Path(args.raw), Path(args.labels)
    rows, total, matched = [], 0, 0
    for path in sorted(raw_directory.glob("*.gdf")):
        codes = session_codes(path)
        truth_path = label_directory / (path.stem + ".mat")
        if not codes or not truth_path.exists():
            # E session 没有事件码，无从比对。
            rows.append(dict(session=path.stem, codes=len(codes),
                             truth=None, identical=None, note="unverifiable"))
            continue
        truth = bcic_truth_labels(truth_path)
        identical = codes == truth
        total += len(codes)
        matched += sum(1 for left, right in zip(codes, truth) if left == right)
        rows.append(dict(session=path.stem, codes=len(codes), truth=len(truth),
                         identical=identical,
                         note=None if identical else "label sequence differs"))
    report = dict(raw=str(raw_directory.resolve()), labels=str(label_directory.resolve()),
                  sessions=len(rows), verified_labels=total, matched_labels=matched,
                  mismatches=[row["session"] for row in rows if row["identical"] is False],
                  rows=rows)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sessions", "verified_labels", "matched_labels", "mismatches")},
                     ensure_ascii=False))
    if report["mismatches"]:
        raise SystemExit(f"标签序列不一致：{report['mismatches'][:5]}")


if __name__ == "__main__":
    main()
