"""
Computational benchmarking for PRISM.

Profiles runtime, memory usage, and scalability.
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict


def run_benchmarks(
    dataset,
    config: dict,
    device: str = "cuda:0",
) -> Dict:
    """Run computational benchmarks.

    Returns:
        Dict with runtime, memory, and scalability metrics
    """
    results = {}

    # 1. Training time benchmark
    print("\n=== Training Time Benchmark ===")
    results["training"] = _benchmark_training(dataset, config, device)

    # 2. Inference time benchmark
    print("\n=== Inference Time Benchmark ===")
    results["inference"] = _benchmark_inference(dataset, config, device)

    # 3. Memory benchmark
    print("\n=== Memory Benchmark ===")
    results["memory"] = _benchmark_memory(config, device)

    # 4. Scalability
    print("\n=== Scalability Benchmark ===")
    results["scalability"] = _benchmark_scalability(dataset, config, device)

    return results


def _benchmark_training(dataset, config, device) -> Dict:
    """Benchmark training wall-clock time."""
    from ..models.encoder import PRISMEncoder
    from ..training.trainer import PRISMTrainer

    batch_size = config.get("batch_size", 256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    encoder = PRISMEncoder(
        n_genes=config.get("n_genes", 2000),
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 12),
    )

    trainer = PRISMTrainer(encoder, config, device=device)

    # Time 5 epochs
    start = time.time()
    for epoch in range(5):
        trainer.train_epoch(loader, epoch)
    elapsed = time.time() - start

    return {
        "time_5_epochs_seconds": elapsed,
        "time_per_epoch_seconds": elapsed / 5,
        "estimated_50_epochs_minutes": (elapsed / 5 * 50) / 60,
        "batch_size": batch_size,
        "n_cells": len(dataset),
    }


def _benchmark_inference(dataset, config, device) -> Dict:
    """Benchmark inference time per cell."""
    from ..models.encoder import PRISMEncoder

    encoder = PRISMEncoder(
        n_genes=config.get("n_genes", 2000),
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 12),
    ).to(device)
    encoder.eval()

    batch_size = config.get("batch_size", 256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Warmup
    with torch.no_grad():
        for batch in loader:
            encoder(batch["expression"].to(device), batch["genotype"].to(device))
            break

    # Benchmark
    total_cells = 0
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            encoder(batch["expression"].to(device), batch["genotype"].to(device))
            total_cells += batch["expression"].shape[0]
    elapsed = time.time() - start

    return {
        "total_cells": total_cells,
        "total_time_seconds": elapsed,
        "time_per_cell_ms": elapsed / total_cells * 1000,
        "throughput_cells_per_second": total_cells / elapsed,
    }


def _benchmark_memory(config, device) -> Dict:
    """Benchmark GPU memory usage."""
    if not torch.cuda.is_available():
        return {"error": "no_gpu"}

    from ..models.encoder import PRISMEncoder

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    encoder = PRISMEncoder(
        n_genes=config.get("n_genes", 2000),
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 12),
    ).to(device)

    model_mem = torch.cuda.max_memory_allocated() / 1e9

    # Forward pass memory
    batch_size = config.get("batch_size", 256)
    dummy_expr = torch.randint(0, 51, (batch_size, config.get("n_genes", 2000))).to(device)
    dummy_geno = torch.zeros(batch_size, dtype=torch.long).to(device)

    torch.cuda.reset_peak_memory_stats()
    z, cls_repr, recon = encoder(dummy_expr, dummy_geno, return_reconstruction=True)
    loss = z.sum() + recon.sum()
    loss.backward()
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    del encoder, dummy_expr, dummy_geno, output
    torch.cuda.empty_cache()

    return {
        "model_memory_gb": model_mem,
        "peak_memory_gb": peak_mem,
        "batch_size": batch_size,
    }


def _benchmark_scalability(dataset, config, device) -> Dict:
    """Benchmark scaling with dataset size."""
    from ..models.encoder import PRISMEncoder
    from ..training.trainer import PRISMTrainer
    from torch.utils.data import Subset

    sizes = [1000, 5000, 10000]
    max_size = len(dataset)
    sizes = [s for s in sizes if s <= max_size]
    if max_size not in sizes:
        sizes.append(max_size)

    results = {}
    for n in sizes:
        indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=min(config.get("batch_size", 256), n),
                           shuffle=True, num_workers=0)

        encoder = PRISMEncoder(
            n_genes=config.get("n_genes", 2000),
            d_model=config.get("d_model", 512),
            n_layers=config.get("n_layers", 12),
        )
        trainer = PRISMTrainer(encoder, config, device=device)

        start = time.time()
        trainer.train_epoch(loader, 0)
        elapsed = time.time() - start

        results[f"n={n}"] = {
            "n_cells": n,
            "time_per_epoch_seconds": elapsed,
        }

        del encoder, trainer
        torch.cuda.empty_cache()

    return results
