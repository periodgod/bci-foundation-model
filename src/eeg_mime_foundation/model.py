"""Small, replaceable EEG/time-series encoders, not paper reproductions."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import Batch


@dataclass
class ModelConfig:
    architecture: str = "fusion"
    sample_rate: int = 64
    patch_samples: int = 32
    dim: int = 48
    depth: int = 2
    heads: int = 4
    mlp_ratio: int = 3
    dropout: float = 0.0
    channel_vocab: int = 128
    latent_channels: int = 4
    local_patches: int = 8
    adapter_dim: int = 12
    task: str = "classification"
    outputs: int = 2
    bands: tuple = ((1., 4.), (4., 8.), (8., 13.), (13., 30.))

    def __post_init__(self):
        if self.architecture not in {"fusion", "factorized", "compact", "latent", "hierarchical", "causal"}:
            raise ValueError("Unknown architecture")
        if min(self.sample_rate, self.patch_samples, self.dim, self.depth, self.heads,
               self.channel_vocab, self.latent_channels, self.local_patches, self.adapter_dim, self.outputs) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.dim % self.heads or self.patch_samples < 4 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid head/patch/dropout setting")
        if self.task not in {"classification", "regression", "sequence_classification", "sequence_regression"}:
            raise ValueError("Unknown task head")
        if not self.bands or any(not 0 <= lo < hi <= self.sample_rate / 2 for lo, hi in self.bands):
            raise ValueError("Frequency bands must be within Nyquist")


def patchify(batch: Batch, size: int):
    if batch.x.ndim != 3 or batch.valid.shape != batch.x.shape:
        raise ValueError("Expected signal/validity [B,C,T]")
    if not batch.valid.flatten(1).any(1).all():
        raise ValueError("All-invalid sample")
    if ((~batch.valid[..., :-1]) & batch.valid[..., 1:]).any():
        raise ValueError("Validity must be right padding; artifact gaps require separate segments")
    if not torch.isfinite(batch.x[batch.valid]).all():
        raise ValueError("Non-finite valid input")
    extra = (-batch.x.shape[-1]) % size
    x = F.pad(batch.x.masked_fill(~batch.valid, 0), (0, extra)).unfold(-1, size, size)
    valid = F.pad(batch.valid, (0, extra)).unfold(-1, size, size)
    return x, valid


def time_position(patches, dim, device, dtype, patch_seconds):
    t = torch.arange(patches, device=device, dtype=dtype) * patch_seconds
    f = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000) / dim))
    phase = t[:, None] * f[None]
    position = torch.zeros(patches, dim, device=device, dtype=dtype)
    position[:, 0::2] = phase.sin()
    position[:, 1::2] = phase[:, :dim // 2].cos()
    return position


def attention(module, query, source, key_valid, query_valid, causal=False):
    """Do not send all-masked rows through softmax. Zero invalid query outputs."""
    good = key_valid.any(-1)
    result = torch.zeros_like(query)
    if good.any():
        mask = None
        if causal:
            if query.shape[1] != source.shape[1]:
                raise ValueError("Causal attention requires matching time axes")
            mask = torch.ones(query.shape[1], source.shape[1], dtype=torch.bool,
                              device=query.device).triu(1)
        attended, _ = module(query[good], source[good], source[good],
                             key_padding_mask=~key_valid[good], attn_mask=mask, need_weights=False)
        result[good] = attended
    return result.masked_fill(~query_valid[..., None], 0)


class FactorBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temporal = nn.MultiheadAttention(cfg.dim, cfg.heads, cfg.dropout, batch_first=True)
        self.spatial = nn.MultiheadAttention(cfg.dim, cfg.heads, cfg.dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.dim) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(cfg.dim, cfg.dim * cfg.mlp_ratio), nn.GELU(),
                                nn.Dropout(cfg.dropout), nn.Linear(cfg.dim * cfg.mlp_ratio, cfg.dim))

    def forward(self, x, valid):
        b, c, p, d = x.shape
        z, m = self.norms[0](x).reshape(b * c, p, d), valid.reshape(b * c, p)
        x = x + attention(self.temporal, z, z, m, m).reshape(b, c, p, d)
        z = self.norms[1](x).transpose(1, 2).reshape(b * p, c, d)
        m = valid.transpose(1, 2).reshape(b * p, c)
        x = x + attention(self.spatial, z, z, m, m).reshape(b, p, c, d).transpose(1, 2)
        return (x + self.ff(self.norms[2](x))).masked_fill(~valid[..., None], 0)


class CompactBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temporal = nn.Conv1d(cfg.dim, cfg.dim, 3, padding=1, groups=cfg.dim)
        self.norm = nn.LayerNorm(cfg.dim)
        self.mix = nn.Sequential(nn.Linear(cfg.dim * 2, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))

    def forward(self, x, valid):
        b, c, p, d = x.shape
        z = self.norm(x).masked_fill(~valid[..., None], 0)
        local = self.temporal(z.reshape(b * c, p, d).transpose(1, 2)).transpose(1, 2).reshape(b, c, p, d)
        context = z.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)[..., None]
        z = self.mix(torch.cat((local, context.expand_as(local)), -1))
        return (x + z).masked_fill(~valid[..., None], 0)


class SignalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        k, d = cfg.patch_samples, cfg.dim
        self.raw = nn.Linear(k, d)
        # Only fusion variants use spectral content; causal uses a completed raw patch.
        self.use_spectrum = cfg.architecture in {"fusion", "latent", "hierarchical"}
        if self.use_spectrum:
            self.wave = nn.Sequential(nn.Conv1d(1, 8, 5, padding=2), nn.GELU(),
                                      nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(8, d))
            self.spectrum = nn.Linear(k // 2 + 1, d)
            self.fusion = nn.Linear(d * 3, d)
        self.ids = nn.Embedding(cfg.channel_vocab, d)
        self.coordinates = nn.Linear(3, d, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.input_norm = nn.LayerNorm(d)
        if cfg.architecture == "compact":
            self.blocks = nn.ModuleList([CompactBlock(cfg) for _ in range(cfg.depth)])
        elif cfg.architecture == "causal":
            self.recurrent = nn.GRU(d, d, cfg.depth, batch_first=True,
                                    dropout=cfg.dropout if cfg.depth > 1 else 0)
        else:
            self.blocks = nn.ModuleList([FactorBlock(cfg) for _ in range(cfg.depth)])
        if cfg.architecture == "latent":
            self.queries = nn.Parameter(torch.randn(cfg.latent_channels, d) * .02)
            self.compress = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
            self.retrieve = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        if cfg.architecture == "hierarchical":
            self.epoch_gru = nn.GRU(d, d, batch_first=True)
        self.adapter = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.adapter_dim), nn.GELU(),
                                     nn.Linear(cfg.adapter_dim, d))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.final_norm = nn.LayerNorm(d)

    def _hierarchical(self, x, valid):
        b, c, p, d = x.shape
        chunk = self.cfg.local_patches
        extra = (-p) % chunk
        z = F.pad(x, (0, 0, 0, extra)).reshape(b, c, -1, chunk, d)
        m = F.pad(valid, (0, extra)).reshape(b, c, -1, chunk)
        n = z.shape[2]
        z = z.permute(0, 2, 1, 3, 4).reshape(b * n, c, chunk, d)
        m = m.permute(0, 2, 1, 3).reshape(b * n, c, chunk)
        for block in self.blocks:
            z = block(z, m)
        pooled = z.sum((1, 2)) / m.sum((1, 2)).clamp_min(1)[:, None]
        context, _ = self.epoch_gru(pooled.reshape(b, n, d))
        z = z.reshape(b, n, c, chunk, d) + context[:, :, None, None]
        return z.permute(0, 2, 1, 3, 4).reshape(b, c, n * chunk, d)[:, :, :p]

    def forward(self, batch, token_mask=None):
        raw, sample_valid = patchify(batch, self.cfg.patch_samples)
        valid, weights = sample_valid.any(-1), sample_valid.float().mean(-1)
        b, c, p, k = raw.shape
        if token_mask is None:
            token_mask = torch.zeros_like(valid)
        if token_mask.shape != valid.shape or (token_mask & ~valid).any():
            raise ValueError("SSL mask must select valid tokens only")
        safe_raw = raw.masked_fill(token_mask[..., None], 0)
        content = self.raw(safe_raw)
        if self.use_spectrum:
            wave = self.wave(safe_raw.reshape(-1, 1, k)).reshape(b, c, p, -1)
            # FFT spacing is sample_rate/patch_samples; this is not long-window SSVEP resolution.
            spectrum = torch.fft.rfft(safe_raw.float(), norm="ortho").abs().to(raw.dtype)
            content = self.fusion(torch.cat((content, wave, self.spectrum(torch.log1p(spectrum))), -1))
        content = torch.where(token_mask[..., None], self.mask_token, content)
        channel_real = valid.any(-1)
        ids = batch.channel_ids
        if (((ids >= self.cfg.channel_vocab) | (ids < -1)) & channel_real).any():
            raise ValueError("Channel ID outside vocabulary")
        if ((ids < 0) & ~batch.coord_valid & channel_real).any():
            raise ValueError("Missing channel identity and coordinates")
        coord = self.coordinates(batch.coords.masked_fill(~batch.coord_valid[..., None], 0))
        identity = self.ids(ids.clamp_min(0)) * (ids >= 0)[..., None]
        position = time_position(p, self.cfg.dim, raw.device, raw.dtype,
                                 self.cfg.patch_samples / self.cfg.sample_rate)
        # Keep channel/time position even for masked patches.
        x = self.input_norm(content + (coord + identity)[:, :, None] + position[None, None])
        x = x.masked_fill(~valid[..., None], 0)
        architecture = self.cfg.architecture
        if architecture == "causal":
            # State resets per independent sample; no bidirectionality, no future context.
            x, _ = self.recurrent(x.reshape(b * c, p, -1))
            x = x.reshape(b, c, p, -1)
        elif architecture == "latent":
            source = x.transpose(1, 2).reshape(b * p, c, -1)
            m = valid.transpose(1, 2).reshape(b * p, c)
            q = self.queries[None].expand(b * p, -1, -1)
            qv = m.any(-1, keepdim=True).expand(-1, self.cfg.latent_channels)
            compressed = attention(self.compress, q, source, m, qv)
            compressed = compressed.reshape(b, p, self.cfg.latent_channels, -1).transpose(1, 2)
            latent_valid = qv.reshape(b, p, self.cfg.latent_channels).transpose(1, 2)
            for block in self.blocks:
                compressed = block(compressed, latent_valid)
            source_latent = compressed.transpose(1, 2).reshape(b * p, self.cfg.latent_channels, -1)
            x = attention(self.retrieve, source, source_latent, qv, m)
            x = x.reshape(b, p, c, -1).transpose(1, 2)
        elif architecture == "hierarchical":
            x = self._hierarchical(x, valid)
        else:
            for block in self.blocks:
                x = block(x, valid)
        x = self.final_norm(x + self.adapter(x)).masked_fill(~valid[..., None], 0)
        return dict(tokens=x, valid=valid, weights=weights, raw=raw, sample_valid=sample_valid)


class SignalModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg, self.encoder = cfg, SignalEncoder(cfg)
        self.head = nn.Linear(cfg.dim, cfg.outputs)
        self.reconstruction = nn.Linear(cfg.dim, cfg.patch_samples)
        self.bandpower = nn.Linear(cfg.dim, len(cfg.bands))

    def forward(self, batch, token_mask=None):
        output = self.encoder(batch, token_mask)
        x, w = output["tokens"], output["weights"]
        sequence = (x * w[..., None]).sum(1) / w.sum(1).clamp_min(1e-8)[..., None]
        sequence_valid = output["valid"].any(1)
        if self.cfg.architecture == "causal":
            index = sequence_valid.sum(-1) - 1
            pooled = sequence[torch.arange(x.shape[0], device=x.device), index]
        else:
            pooled = (x * w[..., None]).sum((1, 2)) / w.sum((1, 2)).clamp_min(1e-8)[:, None]
        output.update(features=pooled, logits=self.head(pooled), sequence_logits=self.head(sequence),
                      sequence_valid=sequence_valid, reconstruction=self.reconstruction(x),
                      bandpower=self.bandpower(x))
        return output

    def configure_tuning(self, strategy):
        if strategy not in {"full", "linear", "adapter"}:
            raise ValueError("Tuning must be full, linear or adapter")
        for name, param in self.named_parameters():
            param.requires_grad_(strategy == "full" or name.startswith("head.") or
                                 (strategy == "adapter" and name.startswith("encoder.adapter.")))

    def config_dict(self):
        return asdict(self.cfg)


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)
