"""可变导联布局的结构不变量。

核心主张是"同一套权重可接受任意通道数的电极布局"。这里把它写成可检验的不变量：
填充通道（valid=False）不得影响输出；整列置为无效则必然改变输出；单通道不得崩溃。
不设分数阈值，只验证结构与数值性质。
"""
import unittest

import torch

from eeg_mime_foundation.data import Batch
from eeg_mime_foundation.model import ModelConfig, SignalModel

T = 560
FULL = 64


def _grid(n: int = FULL) -> torch.Tensor:
    return torch.stack(
        (torch.linspace(-.07, .07, n), torch.zeros(n), torch.full((n,), .08)), dim=1,
    )


def build(n_channels: int = FULL, valid_channels: int | None = None) -> Batch:
    """构造布局；前 FULL 个通道始终是同一批真实电极，多出来的按 collate 的方式补。

    补出的通道 id=-1、坐标沿用已有网格，模拟 collate 的右端填充。
    """
    torch.manual_seed(0)
    base = torch.randn(2, FULL, T)
    if n_channels > FULL:
        x = torch.cat((base, torch.randn(2, n_channels - FULL, T)), dim=1)
    else:
        x = base[:, :n_channels].contiguous()
    valid = torch.ones(2, n_channels, T, dtype=torch.bool)
    if valid_channels is not None and valid_channels < n_channels:
        valid[:, valid_channels:, :] = False
    extra = max(n_channels - FULL, 0)
    real = n_channels - extra
    ids = torch.cat((torch.arange(real), torch.full((extra,), -1, dtype=torch.long)))
    coords = torch.cat((_grid(FULL)[:real], _grid(FULL)[:extra]))
    return Batch(x=x, valid=valid, channel_ids=ids.repeat(2, 1).long(),
                 coords=coords.repeat(2, 1, 1),
                 coord_valid=torch.ones(2, n_channels, dtype=torch.bool),
                 target=torch.tensor([0, 1]), metadata=[{}, {}])


class MontageInvarianceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = ModelConfig(architecture="fusion", sample_rate=160, patch_samples=80,
                               dim=32, depth=2, heads=4, mlp_ratio=3, dropout=0.0,
                               task="classification", outputs=2)
        self.model = SignalModel(self.cfg).eval()

    def test_padding_does_not_change_output(self):
        """尾部补零通道标为无效后，输出必须与短布局一致。"""
        short = self.model(build(FULL))["logits"]
        padded_batch = build(FULL + 16, valid_channels=FULL)
        padded = self.model(padded_batch)["logits"]
        torch.testing.assert_close(padded, short, atol=1e-5, rtol=1e-4)

    def test_dropping_channels_changes_output(self):
        """整列置为无效必须真的作用到前向计算，而不是被静默忽略。"""
        full = self.model(build(FULL))["logits"]
        dropped_batch = build(FULL)
        dropped_batch.valid[:, :32, :] = False
        dropped = self.model(dropped_batch)["logits"]
        self.assertEqual(dropped.shape, full.shape)
        self.assertFalse(torch.allclose(dropped, full, atol=1e-4))

    def test_single_channel_runs(self):
        logits = self.model(build(1))["logits"]
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_sparse_montage_batch_mixes(self):
        """一个 batch 内不同通道数的样本共存（collate 已补齐），输出形状不变。"""
        batch = build(FULL)
        batch.valid[0, 40:, :] = False
        logits = self.model(batch)["logits"]
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
