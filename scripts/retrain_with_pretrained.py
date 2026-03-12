#!/usr/bin/env python
"""Retrain all 15 systems with PCP pre-trained weights and evaluate.

For each system:
1. Load preprocessed adata (existing X_prism from random init is preserved)
2. Train new encoder with pre-trained PCP weights transferred
3. Save new embeddings as X_prism_pretrained in adata
4. Compute metrics for PRISM_pretrained
5. Append to full_evaluation_results.csv

This allows direct comparison: PRISM (random init) vs PRISM (pre-trained).
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anndata as ad
from collections import OrderedDict

# All 15 systems
SYSTEMS = OrderedDict([
    ("skin",           {"adata": "data/processed/adata_processed.h5ad",                   "config": "configs/skin.yaml"}),
    ("pancreas",       {"adata": "data/processed/pancreas/adata_processed.h5ad",           "config": "configs/pancreas.yaml"}),
    ("cortex",         {"adata": "data/processed/cortex/adata_processed.h5ad",             "config": "configs/cortex.yaml"}),
    ("hsc",            {"adata": "data/processed/hsc/adata_processed.h5ad",                "config": "configs/hsc.yaml"}),
    ("cardiac",        {"adata": "data/processed/cardiac/adata_processed.h5ad",            "config": "configs/cardiac.yaml"}),
    ("intestine",      {"adata": "data/processed/intestine/adata_processed.h5ad",          "config": "configs/intestine.yaml"}),
    ("lung",           {"adata": "data/processed/lung/adata_processed.h5ad",               "config": "configs/lung.yaml"}),
    ("neural_crest",   {"adata": "data/processed/neural_crest/adata_processed.h5ad",       "config": "configs/neural_crest.yaml"}),
    ("oligo",          {"adata": "data/processed/oligo/adata_processed.h5ad",              "config": "configs/oligo.yaml"}),
    ("thcell",         {"adata": "data/processed/thcell/adata_processed.h5ad",             "config": "configs/thcell.yaml"}),
    ("paul",           {"adata": "data/processed/paul/adata_processed.h5ad",               "config": "configs/paul.yaml"}),
    ("nestorowa",      {"adata": "data/processed/nestorowa/adata_processed.h5ad",          "config": "configs/nestorowa.yaml"}),
    ("sadefeldman",    {"adata": "data/processed/sadefeldman/adata_processed.h5ad",        "config": "configs/sadefeldman.yaml"}),
    ("tirosh_melanoma",{"adata": "data/processed/tirosh_melanoma/adata_processed.h5ad",    "config": "configs/tirosh_melanoma.yaml"}),
    ("neftel_gbm",     {"adata": "data/processed/neftel_gbm/adata_processed.h5ad",        "config": "configs/neftel_gbm.yaml"}),
])

PRETRAINED_CHECKPOINT = "checkpoints/pretrain/pcp_best.pt"
DEVICE = "cuda:0"


def load_system_config(config_path):
    """Load system config from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sys_cfg = cfg.get("system", cfg)
    condition_key = sys_cfg.get("condition_key", "genotype")
    fate_categories = sys_cfg.get("fate_categories", sys_cfg.get("fate_names", []))
    return condition_key, fate_categories


def train_with_pretrained(system_name, system_info):
    """Train PRISM encoder with pre-trained weights for one system."""
    from prism.data.dataset import PRISMDataset, build_dataloaders
    from prism.data.preprocess import split_data
    from prism.models.encoder import PRISMEncoder
    from prism.training.trainer import PRISMTrainer
    from prism.pretrain.model import PCPEncoder
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print(f"  Training with pre-trained weights: {system_name}")
    print(f"{'='*60}")

    adata_path = system_info["adata"]
    if not os.path.exists(adata_path):
        print(f"  SKIP: {adata_path} not found")
        return None, None

    adata = ad.read_h5ad(adata_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    condition_key, fate_categories = load_system_config(system_info["config"])
    n_fate_categories = len(fate_categories) if fate_categories else 3

    # Determine n_conditions
    n_conditions = 2
    if condition_key in adata.obs.columns:
        n_conditions = max(2, adata.obs[condition_key].nunique())

    # Data split
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_adata, val_adata, test_adata = split_data(adata, seed=seed, condition_key=condition_key)

    n_genes = min(2000, adata.var["highly_variable"].sum() if "highly_variable" in adata.var else 2000)

    train_dataset = PRISMDataset(train_adata, n_genes=n_genes, condition_key=condition_key)
    val_dataset = PRISMDataset(val_adata, n_genes=n_genes, condition_key=condition_key)

    # Scale batch size
    n_gpus = torch.cuda.device_count()
    batch_size = 256 if n_gpus >= 4 else max(16, 32 * n_gpus)
    train_loader, val_loader = build_dataloaders(
        train_dataset, val_dataset,
        batch_size=batch_size, num_workers=0, seed=seed,
    )

    # Load PCP checkpoint to infer d_ff
    ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location="cpu", weights_only=False)
    pcp_d_ff = 2048
    for k, v in ckpt["encoder_state_dict"].items():
        if "ff.0.weight" in k:
            pcp_d_ff = v.shape[0]
            break

    # Build PRISM encoder matching PCP architecture
    encoder = PRISMEncoder(
        n_genes=n_genes,
        n_bins=51,
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_ff=pcp_d_ff,
        d_output=256,
        dropout=0.1,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        n_conditions=n_conditions,
        projection_dims=[512, 256, 128],
    )

    # Transfer PCP weights
    pcp_encoder = PCPEncoder(
        n_genes=n_genes, n_bins=51, d_model=512,
        n_layers=12, n_heads=8, d_ff=pcp_d_ff,
    )
    pcp_encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    transfer_log = pcp_encoder.transfer_weights_to_prism(encoder)
    n_transferred = sum(1 for v in transfer_log.values() if "transferred" in v)
    print(f"  Transferred {n_transferred} parameters from PCP")
    del pcp_encoder, ckpt

    # Train
    config = {
        "device": DEVICE,
        "n_fate_categories": n_fate_categories,
        "lr_lora": 2e-4,
        "lr_head": 1e-3,
        "weight_decay": 0.01,
        "temperature_init": 0.07,
        "alpha_max": 2.0,
        "curriculum_warmup_epochs": 10,
        "info_reg_lambda": 0.1,
        "recon_weight": 0.1,
        "scheduler": "cosine",
        "lr_min_lora": 1e-6,
        "lr_min_head": 1e-5,
        "gradient_clip": 1.0,
    }
    trainer = PRISMTrainer(encoder, config, device=DEVICE)

    checkpoint_dir = f"checkpoints/{system_name}_pretrained"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_result = trainer.train(
        train_loader, val_loader,
        n_epochs=50,
        patience=10,
        checkpoint_dir=checkpoint_dir,
    )
    print(f"  Trained {train_result['n_epochs_trained']} epochs, "
          f"best_val_loss={train_result['best_val_loss']:.4f}")

    # Extract embeddings for ALL cells
    full_dataset = PRISMDataset(adata, n_genes=n_genes, condition_key=condition_key)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    full_embeddings, full_labels, _ = trainer.extract_embeddings(full_loader)

    # Save as X_prism_pretrained
    adata.obsm["X_prism_pretrained"] = full_embeddings
    adata.write_h5ad(adata_path)
    print(f"  Saved X_prism_pretrained: {full_embeddings.shape}")

    return full_embeddings, full_labels


def evaluate_pretrained(system_name, system_info, embeddings, labels):
    """Compute metrics for PRISM_pretrained embeddings."""
    from prism.utils.metrics import compute_extended_metrics

    if embeddings is None:
        return None

    # Handle NaN/Inf
    if np.any(~np.isfinite(embeddings)):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # Get batch labels
    adata = ad.read_h5ad(system_info["adata"])
    condition_key, _ = load_system_config(system_info["config"])
    batch_labels = None
    for key in [condition_key, "sample", "batch", "library", "plate"]:
        if key in adata.obs.columns and adata.obs[key].nunique() > 1:
            batch_labels = adata.obs[key].astype(str).values
            break

    X_original = None
    if hasattr(adata.X, "toarray"):
        if adata.n_obs * adata.n_vars < 500_000_000:
            X_original = adata.X.toarray()
    elif isinstance(adata.X, np.ndarray):
        X_original = adata.X

    try:
        metrics = compute_extended_metrics(
            embeddings, labels,
            batch_labels=batch_labels,
            X_original=X_original,
            method_name="PRISM_pretrained",
        )
        return metrics
    except Exception as e:
        print(f"  Metrics failed: {e}")
        return None


def main():
    import pandas as pd

    print("=" * 60)
    print("  PCP Pre-trained Weight Transfer + Evaluation")
    print(f"  Checkpoint: {PRETRAINED_CHECKPOINT}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Load existing results
    csv_path = "data/processed/full_evaluation_results.csv"
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        print(f"Existing results: {len(df_existing)} rows")
    else:
        df_existing = pd.DataFrame()

    new_rows = []

    for system_name, system_info in SYSTEMS.items():
        # Check if already evaluated
        if len(df_existing) > 0:
            mask = (df_existing["system"] == system_name) & (df_existing["method"] == "PRISM_pretrained")
            if mask.any():
                print(f"\n  {system_name}: PRISM_pretrained already evaluated, skipping")
                continue

        # Train
        embeddings, labels = train_with_pretrained(system_name, system_info)
        if embeddings is None:
            continue

        # Evaluate
        metrics = evaluate_pretrained(system_name, system_info, embeddings, labels)
        if metrics is None:
            continue

        row = {"system": system_name, "method": "PRISM_pretrained"}
        row.update(metrics)
        new_rows.append(row)

        # Print key metrics
        print(f"  {system_name} PRISM_pretrained: "
              f"ARI={metrics.get('ARI', 'N/A'):.4f}, "
              f"RF_AUROC={metrics.get('RF_AUROC', 'N/A'):.4f}")

        # Save incrementally
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)

        # Clear GPU memory
        torch.cuda.empty_cache()

    # Final summary
    print("\n" + "=" * 60)
    print("  PRISM_pretrained vs PRISM (random init) — RF_AUROC")
    print("=" * 60)

    df_final = pd.read_csv(csv_path)
    for sys_name in SYSTEMS:
        prism_row = df_final[(df_final["system"] == sys_name) & (df_final["method"] == "PRISM")]
        pt_row = df_final[(df_final["system"] == sys_name) & (df_final["method"] == "PRISM_pretrained")]
        if len(prism_row) > 0 and len(pt_row) > 0:
            ri = prism_row["RF_AUROC"].values[0]
            pt = pt_row["RF_AUROC"].values[0]
            delta = pt - ri
            arrow = "+" if delta > 0 else ""
            print(f"  {sys_name:20s}: random={ri:.4f}  pretrained={pt:.4f}  delta={arrow}{delta:.4f}")

    print(f"\nDone! Updated CSV: {csv_path}")


if __name__ == "__main__":
    main()
