"""导联布局 → 三维头表坐标（米）。

历史预处理把 `coords` 全部写成 0、`coord_valid` 全部写成 False，于是模型里的
`nn.Linear(3, d)` 坐标编码接收到的永远是零向量——坐标分支存在，但从未被激活。
这个模块是让它真正生效的唯一入口。

坐标来源：MNE 标准电极模板（`standard_1020`，实为 10-10 扩展，94 个导联），
头坐标系原点在头中心，+x 右、+y 前、+z 上，单位米。双极导联取两电极中点。
"""

from __future__ import annotations

import numpy as np

_TEMPLATES: dict[str, dict[str, np.ndarray]] = {}


def _template(template: str) -> dict[str, np.ndarray]:
    if template not in _TEMPLATES:
        import warnings

        import mne

        with warnings.catch_warnings():
            # MNE 1.13 起提示 standard_1020 将在 1.14 更名，模板内容不变。
            warnings.simplefilter("ignore", FutureWarning)
            positions = mne.channels.make_standard_montage(template).get_positions()["ch_pos"]
        _TEMPLATES[template] = {key.upper(): np.asarray(value, dtype=np.float64)
                                for key, value in positions.items()}
    return _TEMPLATES[template]


def normalise(name: str) -> str:
    """导联名归一化：统一大写、去掉空白与常见后缀噪声（`Fp1.` → `FP1`）。"""
    return str(name).strip().strip(".").replace(" ", "").upper()


def standard_coordinates(names, template: str = "standard_1020"):
    """按导联名查表。未收录的名字保留坐标 0 且 `valid=False`，由调用方决定是否接受。"""
    table = _template(template)
    coords = np.zeros((len(names), 3), dtype=np.float32)
    valid = np.zeros(len(names), dtype=bool)
    for index, name in enumerate(names):
        position = table.get(normalise(name))
        if position is not None:
            coords[index] = position.astype(np.float32)
            valid[index] = True
    return coords, valid


def bipolar_coordinates(names, template: str = "standard_1020"):
    """双极导联（`Fpz-Cz`）取两极中点；任一端缺失则整条判为无效。"""
    table = _template(template)
    coords = np.zeros((len(names), 3), dtype=np.float32)
    valid = np.zeros(len(names), dtype=bool)
    for index, name in enumerate(names):
        text = str(name).upper().replace("EEG", "").strip().strip(".").replace(" ", "")
        parts = [part for part in text.split("-") if part]
        if len(parts) != 2:
            continue
        first, second = table.get(parts[0]), table.get(parts[1])
        if first is None or second is None:
            continue
        coords[index] = ((first + second) / 2).astype(np.float32)
        valid[index] = True
    return coords, valid


def coordinates_for(names, kind: str = "scalp_acquisition_channel", template: str = "standard_1020"):
    """按 manifest 的 `channel_kind` 选择解析方式，双极优先走中点。"""
    if kind == "bipolar_derivation":
        return bipolar_coordinates(names, template)
    return standard_coordinates(names, template)


def distance_matrix(coords, valid=None) -> np.ndarray:
    """成对欧氏距离（米）。无效导联的行列置 0，便于直接做掩码。"""
    coords = np.asarray(coords, dtype=np.float64)
    square = (coords[:, None, :] - coords[None, :, :]) ** 2
    distance = np.sqrt(square.sum(-1))
    if valid is not None:
        keep = np.asarray(valid, dtype=bool)
        distance = distance * keep[:, None] * keep[None, :]
    return distance.astype(np.float32)
