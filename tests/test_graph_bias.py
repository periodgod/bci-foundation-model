"""图偏置与导联丢弃增广的不变量。

这两项都是"让模型真的用上电极拓扑"的改动，所以用结构性质而不是分数来验证：
偏置零初始化时与普通 attention 等价；通道置换不改变输出（拓扑由坐标决定，与列顺序无关）；
填充通道仍然不产生任何影响；导联丢弃不会把样本丢空、也不会破坏填充约定。
"""
import unittest

import torch

from eeg_mime_foundation.data import Batch
from eeg_mime_foundation.engine import drop_channels
from eeg_mime_foundation.model import ModelConfig, SignalModel

T = 560
FULL = 32


def _coords(n=FULL):
    angle = torch.linspace(0, 6.28, n)
    return torch.stack((.07 * torch.cos(angle), .07 * torch.sin(angle), torch.full((n,), .05)), 1)


def build(n_channels=FULL, valid_channels=None):
    torch.manual_seed(3)
    base = torch.randn(2, FULL, T)
    extra = max(n_channels - FULL, 0)
    # 先在固定批上取样，再追加填充通道，保证两种布局的前 FULL 通道逐位相同。
    x = torch.cat((base, torch.randn(2, extra, T)), 1) if extra else base[:, :n_channels].contiguous()
    valid = torch.ones(2, n_channels, T, dtype=torch.bool)
    if valid_channels is not None and valid_channels < n_channels:
        valid[:, valid_channels:, :] = False
    coords = torch.cat((_coords(FULL), _coords(FULL)[:extra])) if extra else _coords(n_channels)
    return Batch(x=x, valid=valid,
                 channel_ids=torch.arange(n_channels).repeat(2, 1).clamp(max=n_channels - 1).long(),
                 coords=coords.repeat(2, 1, 1),
                 coord_valid=torch.ones(2, n_channels, dtype=torch.bool),
                 target=torch.tensor([0, 1]), metadata=[{}, {}])


class GraphBiasTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = ModelConfig(architecture="fusion", sample_rate=160, patch_samples=80,
                               dim=32, depth=2, heads=4, task="classification", outputs=2,
                               graph=True, graph_bins=6)
        self.model = SignalModel(self.cfg).eval()

    def test_zero_bias_at_init(self):
        """零初始化：图偏置在起点必须恒为 0，训练起点与普通 attention 完全等价。"""
        batch = build()
        for block in self.model.encoder.blocks:
            bias = block.distance_bias(batch.coords, batch.coord_valid)
            self.assertEqual(float(bias.abs().max()), 0.0)

    def test_padding_does_not_change_output(self):
        short = self.model(build(FULL))["logits"]
        padded = self.model(build(FULL + 8, valid_channels=FULL))["logits"]
        torch.testing.assert_close(padded, short, atol=1e-5, rtol=1e-4)

    def test_channel_permutation_is_equivariant(self):
        """打乱通道顺序（信号/ID/坐标同步打乱）不得改变输出：拓扑由坐标决定。"""
        batch = build()
        order = torch.randperm(FULL, generator=torch.Generator().manual_seed(7))
        shuffled = Batch(batch.x[:, order], batch.valid[:, order], batch.channel_ids[:, order],
                         batch.coords[:, order], batch.coord_valid[:, order], batch.target,
                         batch.metadata)
        torch.testing.assert_close(self.model(shuffled)["logits"], self.model(batch)["logits"],
                                   atol=1e-4, rtol=1e-4)

    def test_bias_responds_to_distance(self):
        """偏置必须真的由距离决定：改变偏置权重后，近邻与远端的偏置不再相同且保持对称。"""
        block = self.model.encoder.blocks[0]
        torch.nn.init.normal_(block.bias.weight, std=.5)
        batch = build()
        bias = block.distance_bias(batch.coords, batch.coord_valid)
        self.assertGreater(float((bias - bias[:, :, :1, :]).abs().max()), 1e-4)
        torch.testing.assert_close(bias, bias.transpose(-1, -2), atol=1e-4, rtol=1e-3)

    def test_missing_coordinates_zero_the_bias(self):
        block = self.model.encoder.blocks[0]
        torch.nn.init.normal_(block.bias.weight, std=.5)
        batch = build()
        batch.coord_valid[:, 5:] = False
        bias = block.distance_bias(batch.coords, batch.coord_valid)
        self.assertEqual(float(bias[:, :, 5:, :].abs().max()), 0.0)

    def test_graph_rejects_latent(self):
        with self.assertRaises(ValueError):
            ModelConfig(architecture="latent", graph=True)


class ChannelDropoutTests(unittest.TestCase):
    def test_keeps_at_least_one_channel_and_respects_padding(self):
        batch = build(FULL)
        batch.valid[1, 24:, :] = False          # 第二个样本本来就只有 24 个有效通道
        dropped = drop_channels(batch, .9, generator=torch.Generator().manual_seed(0))
        self.assertTrue(dropped.valid.any(-1).any(-1).all())
        self.assertFalse((dropped.valid & ~batch.valid).any())
        self.assertGreater(float(dropped.valid[1, :24].float().mean()), 0.)

    def test_zero_rate_is_identity(self):
        batch = build()
        self.assertIs(drop_channels(batch, 0.), batch)


if __name__ == "__main__":
    unittest.main()
