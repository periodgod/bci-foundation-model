"""Causal reference inference: completed patches only, explicit recording reset.

Recomputes the prefix to keep the interface simple and verifiable. This is NOT
an optimized bounded-memory streaming GRU implementation. Measure latency first.
"""
import torch

from .data import Batch


class CausalStream:
    def __init__(self, model, normalizer):
        if model.cfg.architecture != "causal":
            raise ValueError("Streaming requires the causal architecture")
        self.model, self.normalizer = model.eval(), normalizer
        self.reset()

    def reset(self):
        self.buffer, self.channels, self.emitted = None, None, 0

    @torch.no_grad()
    def push(self, chunk, channel_ids, coords, coord_valid):
        if chunk.ndim != 2 or chunk.shape[0] != len(channel_ids) or not torch.isfinite(chunk).all():
            raise ValueError("Finite streaming chunk [C,T] required")
        channels = (channel_ids.clone(), coords.clone(), coord_valid.clone())
        if self.channels is not None and any(not torch.equal(a, b) for a, b in zip(channels, self.channels)):
            raise ValueError("Channel layout changed; reset at a recording boundary")
        self.channels = channels
        self.buffer = chunk.clone() if self.buffer is None else torch.cat((self.buffer, chunk), -1)
        complete = self.buffer.shape[-1] // self.model.cfg.patch_samples
        if complete == self.emitted:
            return chunk.new_empty((0, self.model.cfg.outputs))
        x = self.buffer[:, :complete * self.model.cfg.patch_samples][None]
        batch = Batch(x, torch.ones_like(x, dtype=torch.bool), channel_ids[None], coords[None],
                      coord_valid[None], None, [])
        output = self.model(self.normalizer.apply(batch))["sequence_logits"][0, self.emitted:complete]
        self.emitted = complete
        return output
