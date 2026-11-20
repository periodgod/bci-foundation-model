"""SSL and multi-teacher objectives with explicit alignment and applicability."""
import copy

import torch
from torch import nn
from torch.nn import functional as F


def make_mask(valid, ratio, kind="joint", generator=None):
    if not 0 < ratio < 1 or kind not in {"random", "time", "channel", "joint"}:
        raise ValueError("Mask ratio must be (0,1), kind random/time/channel/joint")
    b, c, p = valid.shape
    shape = {"random": (b, c, p), "time": (b, 1, p), "channel": (b, c, 1), "joint": (b, c, p)}[kind]
    if kind == "joint":
        r = 1 - (1 - ratio) ** .5
        mask = ((torch.rand((b, c, 1), device=valid.device, generator=generator) < r) |
                (torch.rand((b, 1, p), device=valid.device, generator=generator) < r)).expand_as(valid).clone()
    else:
        mask = (torch.rand(shape, device=valid.device, generator=generator) < ratio).expand_as(valid).clone()
    mask &= valid
    # Guarantee at least one target and one visible token; structure may be relaxed at these boundaries.
    for i in range(b):
        indices = valid[i].flatten().nonzero().flatten()
        if indices.numel() < 2:
            raise ValueError("SSL requires at least two valid tokens per sample")
        flat = mask[i].view(-1)
        if not flat[indices].any():
            flat[indices[0]] = True
        if flat[indices].all():
            flat[indices[-1]] = False
    return mask


def masked_average(values, mask):
    if not mask.any():
        raise ValueError("Loss has no valid targets")
    return values.masked_select(mask).mean()


def band_targets(raw, sample_valid, sample_rate, bands):
    # Partial patches are excluded from spectral targets, not treated as complete observations.
    power = torch.fft.rfft(raw.float(), norm="ortho").abs().square()
    frequency = torch.fft.rfftfreq(raw.shape[-1], d=1 / sample_rate).to(raw.device)
    values = []
    for lo, hi in bands:
        bins = (frequency >= lo) & (frequency < hi)
        if not bins.any():
            raise ValueError(f"Patch FFT cannot resolve band [{lo}, {hi}); increase patch_samples")
        values.append(torch.log(power[..., bins].mean(-1).clamp_min(1e-8)))
    return torch.stack(values, -1), sample_valid.all(-1)


def reconstruction_loss(output, mask, cfg, band_weight=0., spectral_weight=0.):
    selected = mask[..., None] & output["sample_valid"]
    waveform = masked_average((output["reconstruction"] - output["raw"]).square(), selected)
    total, parts = waveform, {"waveform": waveform.detach()}
    complete = mask & output["sample_valid"].all(-1)
    if band_weight:
        target, _ = band_targets(output["raw"], output["sample_valid"], cfg.sample_rate, cfg.bands)
        band = masked_average((output["bandpower"] - target).square(), complete[..., None].expand_as(target))
        total = total + band_weight * band
        parts["bandpower"] = band.detach()
    if spectral_weight:
        predicted = torch.fft.rfft(output["reconstruction"].float(), norm="ortho").abs()
        target = torch.fft.rfft(output["raw"].float(), norm="ortho").abs()
        spectral = masked_average((torch.log1p(predicted) - torch.log1p(target)).square(),
                                  complete[..., None].expand_as(target))
        total = total + spectral_weight * spectral
        parts["spectrum"] = spectral.detach()
    return total, parts


class EMATarget(nn.Module):
    def __init__(self, encoder, decay=.99):
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be [0,1)")
        self.encoder, self.decay = copy.deepcopy(encoder).eval(), decay
        self.requires_grad_(False)

    @torch.no_grad()
    def update(self, encoder):
        for dst, src in zip(self.encoder.parameters(), encoder.parameters()):
            dst.lerp_(src, 1 - self.decay)
        for dst, src in zip(self.encoder.buffers(), encoder.buffers()):
            dst.copy_(src)

    @torch.no_grad()
    def forward(self, batch):
        self.encoder.eval()
        return self.encoder(batch)["tokens"]


def latent_loss(student, teacher, mask):
    error = 1 - F.cosine_similarity(student, teacher.detach(), dim=-1)
    return masked_average(error, mask)


def contrastive_loss(first, second, temperature=.2):
    if first.shape[0] < 2 or temperature <= 0:
        raise ValueError("Contrastive learning requires >=2 samples and positive temperature")
    first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class MultiTeacherLoss(nn.Module):
    """Independent student->teacher projections; embeddings are never averaged.

    applicability [B,K] is externally supplied from verified task/input contracts.
    learned weights require an entropy term and comparison to equal/fixed weights.
    """
    def __init__(self, student_dim, teacher_dims, weights=None, learned=False):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(student_dim, d) for d in teacher_dims])
        w = torch.tensor(weights if weights is not None else [1.] * len(teacher_dims), dtype=torch.float32)
        if not len(teacher_dims) or w.shape != (len(teacher_dims),) or not torch.isfinite(w).all() or (w <= 0).any():
            raise ValueError("Positive weight per teacher required")
        self.register_buffer("weights", w / w.sum())
        self.gate = nn.Linear(student_dim, len(teacher_dims)) if learned else None

    def forward(self, student, teachers, applicability=None, entropy_weight=.01):
        if len(teachers) != len(self.projections):
            raise ValueError("Teacher count mismatch")
        losses = []
        for projection, teacher in zip(self.projections, teachers):
            if teacher.ndim != 2 or teacher.shape[0] != student.shape[0]:
                raise ValueError("Teacher feature batch mismatch")
            losses.append(1 - F.cosine_similarity(projection(student), teacher.detach(), dim=-1))
        errors = torch.stack(losses, -1)
        applicable = torch.ones_like(errors, dtype=torch.bool) if applicability is None else applicability
        if applicable.shape != errors.shape or not applicable.any(-1).all():
            raise ValueError("Each sample needs at least one applicable teacher")
        if self.gate is not None:
            if entropy_weight <= 0:
                raise ValueError("Learned gating requires positive entropy regularization")
            scores = self.gate(student).masked_fill(~applicable, float("-inf"))
            weights = scores.softmax(-1)
        else:
            weights = self.weights[None] * applicable
            weights = weights / weights.sum(-1, keepdim=True)
        loss = (errors * weights).sum(-1).mean()
        if self.gate is not None:
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean()
            loss = loss - entropy_weight * entropy
        return loss, weights.detach()


def logit_distillation(student_logits, teacher_logits, student_labels, teacher_labels, temperature=2., reduction="batchmean"):
    if student_labels != teacher_labels or student_logits.shape != teacher_logits.shape:
        raise ValueError("Output KD requires identical ordered label semantics and shapes")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    loss = F.kl_div(F.log_softmax(student_logits / temperature, -1),
                    F.softmax(teacher_logits.detach() / temperature, -1),
                    reduction="none").sum(-1) * temperature ** 2
    if reduction == "none":
        return loss
    if reduction != "batchmean":
        raise ValueError("KL reduction must be none/batchmean")
    return loss.mean()
