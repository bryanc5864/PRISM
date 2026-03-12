#!/usr/bin/env python3
"""Streamlined ablation study for PRISM.

Runs key ablation variants with reduced epochs to demonstrate
component contributions within a reasonable time.
"""

import os
import sys
import time
import copy
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import warnings
warnings.filterwarnings("ignore")

import torch
import yaml
import anndata as ad
from torch.utils.data import DataLoader

from prism.data.dataset import PRISMDataset, build_dataloaders
from prism.data.preprocess import split_data
from prism.models.encoder import PRISMEncoder
from prism.training.trainer import PRISMTrainer
from prism.utils.metrics import compute_all_metrics


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    flat = {}
    for section in config.values():
        if isinstance(section, dict):
            flat.update(section)
    flat["_structured"] = config
    return flat


def build_encoder(config):
    return PRISMEncoder(
        n_genes=config.get("n_genes", 2000),
        n_bins=config.get("n_expression_bins", 51),
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 12),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 1024),
        d_output=config.get("d_output", 256),
        dropout=config.get("dropout", 0.1),
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        projection_dims=config.get("projection_dims", [512, 256, 128]),
    )


def run_single_ablation(name, config, train_loader, val_loader, test_loader, test_labels, n_epochs=10):
    """Train and evaluate a single ablation variant."""
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        encoder = build_encoder(config)
        trainer = PRISMTrainer(encoder, config, device="cuda:0")
        train_result = trainer.train(
            train_loader, val_loader,
            n_epochs=n_epochs,
            patience=max(5, n_epochs // 2),
            checkpoint_dir=f"checkpoints/ablation_{name.replace('/', '_')}",
        )

        embeddings, labels, genotypes = trainer.extract_embeddings(test_loader)
        metrics = compute_all_metrics(embeddings, test_labels, method_name=name)
        metrics["training_time"] = time.time() - start
        metrics["n_epochs_trained"] = train_result.get("n_epochs_trained", 0)
        metrics["best_val_loss"] = train_result.get("best_val_loss", float("inf"))

        print(f"  ARI={metrics.get('ARI', 0):.4f}, AMI={metrics.get('AMI', 0):.4f}, "
              f"RF_AUROC={metrics.get('RF_AUROC', 0):.4f}, time={metrics['training_time']:.0f}s")
        return metrics

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"method": name, "error": str(e), "training_time": time.time() - start}


def main():
    config = load_config()
    config["device"] = "cuda:0"
    n_epochs = 10  # Reduced epochs for ablation

    print("Loading data...")
    adata = ad.read_h5ad("data/processed/adata_processed.h5ad")
    train_adata, val_adata, test_adata = split_data(adata, seed=42)

    n_genes = min(config.get("n_genes", 2000),
                  adata.var["highly_variable"].sum() if "highly_variable" in adata.var else 2000)
    config["n_genes"] = n_genes

    train_ds = PRISMDataset(train_adata, n_genes=n_genes)
    val_ds = PRISMDataset(val_adata, n_genes=n_genes)
    test_ds = PRISMDataset(test_adata, n_genes=n_genes)

    batch_size = config.get("batch_size", 256)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=batch_size, num_workers=0, seed=42)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    test_labels = test_ds.fate_label

    # Define ablation variants
    ablations = {}

    # 1. Full model (reference)
    ablations["PRISM-full"] = copy.deepcopy(config)

    # 2. Without hard-negatives (α=0)
    c = copy.deepcopy(config)
    c["alpha_max"] = 0.0
    ablations["w/o hard-neg"] = c

    # 3. Without curriculum (α=α_max from start)
    c = copy.deepcopy(config)
    c["curriculum_warmup_epochs"] = 0
    ablations["w/o curriculum"] = c

    # 4. Without info regularizer
    c = copy.deepcopy(config)
    c["info_reg_lambda"] = 0.0
    ablations["w/o info-reg"] = c

    # 5. Without reconstruction loss
    c = copy.deepcopy(config)
    c["recon_weight"] = 0.0
    ablations["w/o recon"] = c

    # 6. Without condition embedding
    c = copy.deepcopy(config)
    c["no_condition"] = True
    ablations["w/o condition"] = c

    # Run all ablations
    results = {}
    total_start = time.time()

    for name, ablation_config in ablations.items():
        results[name] = run_single_ablation(
            name, ablation_config, train_loader, val_loader,
            test_loader, test_labels, n_epochs=n_epochs,
        )
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Write results
    print(f"\n{'='*60}")
    print(f"Ablation Study Complete ({total_time:.0f}s = {total_time/60:.1f}min)")
    print(f"{'='*60}")

    result_text = "**Ablation Study (Experiment 2)**\n\n"
    result_text += "| Variant | ARI | AMI | ASW | RF F1 | RF AUROC | Time(s) |\n"
    result_text += "|---------|-----|-----|-----|-------|----------|------|\n"

    for name, metrics in results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            result_text += (
                f"| {name} | "
                f"{metrics.get('ARI', 0):.3f} | "
                f"{metrics.get('AMI', 0):.3f} | "
                f"{metrics.get('ASW', 0):.3f} | "
                f"{metrics.get('RF_F1_macro', 0):.3f} | "
                f"{metrics.get('RF_AUROC', 0):.3f} | "
                f"{metrics.get('training_time', 0):.0f} |\n"
            )
        else:
            result_text += f"| {name} | FAILED | | | | | |\n"

    print(result_text)

    # Append to results.md
    with open("results.md", "a") as f:
        f.write(f"\n\n---\n\n### Experiment 2: Ablation Studies\n")
        f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(result_text)
        f.write(f"\nTotal ablation time: {total_time:.0f}s ({total_time/60:.1f}min)\n")


if __name__ == "__main__":
    main()
