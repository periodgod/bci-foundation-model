from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class BSGMIMEConfig:
    input_seconds: float = 4.0
    sampling_rate: int = 160
    max_channels: int = 64
    patch_seconds: float = 0.5
    patch_stride_seconds: float = 0.5
    d_model: int = 192
    depth: int = 6
    mlp_ratio: int = 2
    graph_topk: int = 12
    propagation_steps: int = 3
    dropout: float = 0.1
    state_classes: int = 2
    body_classes: int = 4
    joint_classes: int = 8

    @property
    def input_samples(self) -> int:
        return int(round(self.input_seconds * self.sampling_rate))

    @property
    def patch_samples(self) -> int:
        return int(round(self.patch_seconds * self.sampling_rate))

    @property
    def patch_stride_samples(self) -> int:
        return int(round(self.patch_stride_seconds * self.sampling_rate))

    @property
    def num_time_patches(self) -> int:
        return 1 + (self.input_samples - self.patch_samples) // self.patch_stride_samples


MIMEModelConfig = BSGMIMEConfig


class BandAwarePatchEmbedding(nn.Module):
    """Raw patch projection plus fixed mu/beta bandpower cues for MI/ME."""

    def __init__(self, config: BSGMIMEConfig) -> None:
        super().__init__()
        self.config = config
        self.raw_projection = nn.Linear(config.patch_samples, config.d_model)
        self.band_projection = nn.Linear(3, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, channels, samples], got {tuple(x.shape)}")
        if x.shape[-1] != self.config.input_samples:
            raise ValueError(
                f"Expected {self.config.input_samples} samples, got {x.shape[-1]}"
            )

        patches = x.unfold(
            dimension=-1,
            size=self.config.patch_samples,
            step=self.config.patch_stride_samples,
        )
        raw_tokens = self.raw_projection(patches)
        band_tokens = self.band_projection(self._bandpower_features(patches))
        return self.norm(raw_tokens + band_tokens), patches

    def _bandpower_features(self, patches: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(patches.float(), dim=-1)
        power = spectrum.abs().square()
        freqs = torch.fft.rfftfreq(
            self.config.patch_samples,
            d=1.0 / float(self.config.sampling_rate),
        ).to(patches.device)
        bands = [(8.0, 12.0), (13.0, 20.0), (20.0, 30.0)]
        values = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            if torch.any(mask):
                values.append(torch.log1p(power[..., mask].mean(dim=-1)))
            else:
                values.append(torch.zeros_like(power[..., 0]))
        return torch.stack(values, dim=-1).to(dtype=patches.dtype)


class BalancedSignedGraphUnrollBlock(nn.Module):
    """BGL + LPF algorithm-unrolling block inspired by balanced signed graph EEG Transformer."""

    def __init__(self, config: BSGMIMEConfig) -> None:
        super().__init__()
        self.config = config
        self.graph_norm = nn.LayerNorm(config.d_model)
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.sign_head = nn.Linear(config.d_model, 1)
        self.cutoff_logit = nn.Parameter(torch.tensor(0.0))
        self.output_projection = nn.Linear(config.d_model, config.d_model)

        self.temporal_norm = nn.LayerNorm(config.d_model)
        self.temporal_filter = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            padding=1,
            groups=config.d_model,
        )
        hidden = config.d_model * config.mlp_ratio
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, channel_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.output_projection(self._balanced_signed_lpf(self.graph_norm(x), channel_mask))
        x = x + self._temporal_lpf(self.temporal_norm(x))
        return x + self.ffn(self.ffn_norm(x))

    def _balanced_signed_lpf(self, x: Tensor, channel_mask: Optional[Tensor]) -> Tensor:
        batch, channels, patches, dim = x.shape
        nodes = x.permute(0, 2, 1, 3).reshape(batch * patches, channels, dim)
        query = F.normalize(self.query(nodes), dim=-1)
        affinity = torch.matmul(query, query.transpose(-1, -2)) / (dim**0.5)

        valid_mask = None
        if channel_mask is not None:
            valid = channel_mask.bool().repeat_interleave(patches, dim=0)
            valid_mask = valid[:, :, None] & valid[:, None, :]
            affinity = affinity.masked_fill(~valid_mask, -1.0e4)

        if 0 < self.config.graph_topk < channels:
            topk_idx = affinity.topk(k=self.config.graph_topk, dim=-1).indices
            topk_mask = torch.zeros_like(affinity, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            if valid_mask is not None:
                topk_mask = topk_mask & valid_mask
            affinity = affinity.masked_fill(~topk_mask, -1.0e4)

        graph = torch.softmax(affinity, dim=-1)
        polarity = torch.tanh(self.sign_head(nodes))
        balanced_nodes = nodes * polarity
        filtered = balanced_nodes
        keep = torch.sigmoid(self.cutoff_logit)
        for _ in range(self.config.propagation_steps):
            filtered = keep * balanced_nodes + (1.0 - keep) * torch.matmul(graph, filtered)
        restored = filtered * polarity
        return restored.reshape(batch, patches, channels, dim).permute(0, 2, 1, 3)

    def _temporal_lpf(self, x: Tensor) -> Tensor:
        batch, channels, patches, dim = x.shape
        seq = x.reshape(batch * channels, patches, dim).transpose(1, 2)
        filtered = self.temporal_filter(seq).transpose(1, 2)
        return filtered.reshape(batch, channels, patches, dim)


class BalancedSignedGraphMIME(nn.Module):
    """Lightweight MI/ME encoder replacing attention Transformer with BSG unrolling."""

    def __init__(self, config: Optional[BSGMIMEConfig] = None) -> None:
        super().__init__()
        self.config = config or BSGMIMEConfig()
        cfg = self.config

        self.patch_embed = BandAwarePatchEmbedding(cfg)
        self.channel_embedding = nn.Parameter(torch.zeros(1, cfg.max_channels, 1, cfg.d_model))
        self.time_embedding = nn.Parameter(torch.zeros(1, 1, cfg.num_time_patches, cfg.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.d_model))
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([BalancedSignedGraphUnrollBlock(cfg) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(cfg.d_model)

        self.reconstruction_head = nn.Linear(cfg.d_model, cfg.patch_samples)
        self.state_head = nn.Linear(cfg.d_model, cfg.state_classes)
        self.body_head = nn.Linear(cfg.d_model, cfg.body_classes)
        self.joint_head = nn.Linear(cfg.d_model, cfg.joint_classes)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward_features(
        self,
        x: Tensor,
        channel_mask: Optional[Tensor] = None,
        patch_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tokens, _ = self.patch_embed(x)
        _, channels, patches, _ = tokens.shape
        if channels > self.config.max_channels:
            raise ValueError(f"Expected at most {self.config.max_channels} channels, got {channels}")

        tokens = tokens + self.channel_embedding[:, :channels] + self.time_embedding[:, :, :patches]
        if patch_mask is not None:
            tokens = torch.where(patch_mask[..., None].bool(), self.mask_token, tokens)
        tokens = self.dropout(tokens)

        for block in self.blocks:
            tokens = block(tokens, channel_mask=channel_mask)
        return self.final_norm(tokens)

    def forward(
        self,
        x: Tensor,
        channel_mask: Optional[Tensor] = None,
        patch_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        features = self.forward_features(x, channel_mask=channel_mask, patch_mask=patch_mask)
        pooled = self._masked_pool(features, channel_mask)
        return {
            "features": features,
            "reconstruction": self.reconstruction_head(features),
            "state_logits": self.state_head(pooled),
            "body_logits": self.body_head(pooled),
            "joint_logits": self.joint_head(pooled),
        }

    def reconstruction_loss(
        self,
        x: Tensor,
        outputs: Dict[str, Tensor],
        patch_mask: Optional[Tensor] = None,
    ) -> Tensor:
        _, target_patches = self.patch_embed(x)
        loss = F.mse_loss(outputs["reconstruction"], target_patches, reduction="none").mean(dim=-1)
        if patch_mask is not None:
            weights = patch_mask.float()
            return (loss * weights).sum() / weights.sum().clamp_min(1.0)
        return loss.mean()

    def _masked_pool(self, features: Tensor, channel_mask: Optional[Tensor]) -> Tensor:
        if channel_mask is None:
            return features.mean(dim=(1, 2))
        weights = channel_mask.float()[:, :, None, None]
        numerator = (features * weights).sum(dim=(1, 2))
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return numerator / denominator


MIMEFactorizedTransformer = BalancedSignedGraphMIME


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return sum(p.numel() for p in params)
