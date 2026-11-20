"""Measure one real batch at the configured size; parameter counts are not memory."""
import time

import torch

from .data import ManifestDataset, Normalizer, collate
from .engine import set_seed, validate_config
from .model import SignalModel, count_parameters


def profile(config, manifest, device="cpu", repeats=5):
    if repeats < 1:
        raise ValueError("Positive profiling repeats required")
    cfg = validate_config(config)
    torch.set_num_threads(config["training"].get("threads", 2))
    set_seed(config["training"].get("seed", 0))
    dataset = ManifestDataset(manifest, "train", cfg.sample_rate, cfg.task)
    normalizer = Normalizer.fit(dataset)
    n = min(config["training"]["batch_size"], len(dataset))
    raw = collate([dataset[i] for i in range(n)]).to(device)
    batch = normalizer.apply(raw)
    model = SignalModel(cfg).to(device).eval()
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)
    with torch.no_grad():
        model(batch)
        sync()
        started = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        sync()
    milliseconds = (time.perf_counter() - started) * 1000 / repeats
    inference_memory = torch.cuda.max_memory_allocated(device) / 2**20 if is_cuda else None
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    diagnostic_loss = output["logits"].square().mean() + output["reconstruction"].square().mean() + output["bandpower"].square().mean()
    diagnostic_loss.backward()
    optimizer.step()
    sync()
    return dict(device=device, batch_shape=list(batch.x.shape), parameters=count_parameters(model),
                inference_batch_ms=milliseconds, inference_peak_allocated_mib=inference_memory,
                training_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if is_cuda else None,
                objective="diagnostic_all_heads_fp32", teachers_included=False, repeats=repeats,
                scope="This batch only; CUDA allocated memory excludes driver/other processes; CPU memory not measured")
