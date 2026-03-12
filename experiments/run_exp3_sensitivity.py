#!/usr/bin/env python3
"""
Experiment 3: Sensitivity Analysis for PRISM.

Tests robustness of PRISM embeddings along four axes WITHOUT retraining the model:

1. Downsampling robustness: RF AUROC when training classifier on subsampled cells
2. Gene dropout robustness: RF AUROC when dropping embedding dimensions from X_prism
3. Label noise robustness: RF AUROC when flipping eccrine<->hair labels
4. Hyperparameter sensitivity: Document the training configuration

All experiments use 5 random seeds and report mean +/- std.
Results are appended to results.md.
"""

import os
import sys
import time
import warnings
import numpy as np
import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

# Fix library path for torch
import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad")
CONFIG_PATH = os.path.join(PROJECT_DIR, "configs", "default.yaml")
RESULTS_PATH = os.path.join(PROJECT_DIR, "results.md")
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]
RF_ESTIMATORS = 100
TEST_SIZE = 0.3  # held-out fraction for RF evaluation


def load_config(path=CONFIG_PATH):
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rf_auroc(X_train, y_train, X_test, y_test, seed=42):
    """Train RF and return AUROC on the test set."""
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_test)
    # Make sure we get the column for the positive class (eccrine=1)
    if rf.classes_.shape[0] == 2:
        pos_idx = list(rf.classes_).index(1)
        return roc_auc_score(y_test, probs[:, pos_idx])
    else:
        return roc_auc_score(y_test, probs, multi_class="ovr")


def get_labeled_data(adata):
    """Extract X_prism and binary labels for labeled cells (fate_int >= 2)."""
    mask = adata.obs["fate_int"].values >= 2
    X = adata.obsm["X_prism"][mask]
    y_raw = adata.obs["fate_int"].values[mask]
    # Binary: eccrine (2) -> 1, hair (3) -> 0
    y = (y_raw == 2).astype(int)
    return X, y, mask


# ===================================================================
# Experiment 3a: Downsampling Robustness
# ===================================================================
def exp_downsampling(adata):
    """
    Evaluate how RF AUROC degrades when using only a fraction of labeled
    cells for training the classifier. The PRISM embeddings are fixed;
    we subsample the TRAINING split only.
    """
    print("\n" + "=" * 60)
    print("Experiment 3a: Downsampling Robustness")
    print("=" * 60)

    X_all, y_all, _ = get_labeled_data(adata)
    fractions = [1.0, 0.75, 0.50, 0.25, 0.10]
    results = {}

    for frac in fractions:
        aurocs = []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            # Fixed train/test split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
            train_idx, test_idx = next(sss.split(X_all, y_all))

            X_test, y_test = X_all[test_idx], y_all[test_idx]

            # Subsample TRAINING set
            if frac < 1.0:
                n_sub = max(10, int(len(train_idx) * frac))
                sub_idx = rng.choice(train_idx, size=n_sub, replace=False)
            else:
                sub_idx = train_idx

            X_train, y_train = X_all[sub_idx], y_all[sub_idx]

            # Ensure both classes present
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            auc = rf_auroc(X_train, y_train, X_test, y_test, seed=seed)
            aurocs.append(auc)

        pct = int(frac * 100)
        n_train = int(len(train_idx) * frac) if frac < 1.0 else len(train_idx)
        results[pct] = {
            "mean": np.mean(aurocs),
            "std": np.std(aurocs),
            "n_train": n_train,
            "n_runs": len(aurocs),
        }
        print(f"  {pct:>3d}% training cells ({n_train:>4d} cells): "
              f"AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}  "
              f"[{len(aurocs)} seeds]")

    return results


# ===================================================================
# Experiment 3b: Embedding Dimension Dropout
# ===================================================================
def exp_dimension_dropout(adata):
    """
    Measure how RF AUROC changes when randomly zeroing out dimensions
    of the 128-d X_prism embedding. This simulates information loss /
    gene dropout at the representation level.
    """
    print("\n" + "=" * 60)
    print("Experiment 3b: Embedding Dimension Dropout")
    print("=" * 60)

    X_all, y_all, _ = get_labeled_data(adata)
    n_dims = X_all.shape[1]
    drop_fracs = [0.0, 0.10, 0.20, 0.50]
    results = {}

    for drop_frac in drop_fracs:
        aurocs = []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
            train_idx, test_idx = next(sss.split(X_all, y_all))

            # Randomly drop dimensions (same dims for train and test so it is
            # as if the model never computed those dimensions)
            if drop_frac > 0:
                n_drop = int(n_dims * drop_frac)
                drop_dims = rng.choice(n_dims, size=n_drop, replace=False)
                X_dropped = X_all.copy()
                X_dropped[:, drop_dims] = 0.0
            else:
                X_dropped = X_all

            X_train, y_train = X_dropped[train_idx], y_all[train_idx]
            X_test, y_test = X_dropped[test_idx], y_all[test_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            auc = rf_auroc(X_train, y_train, X_test, y_test, seed=seed)
            aurocs.append(auc)

        pct = int(drop_frac * 100)
        n_kept = n_dims - int(n_dims * drop_frac)
        results[pct] = {
            "mean": np.mean(aurocs),
            "std": np.std(aurocs),
            "n_dims_kept": n_kept,
            "n_dims_dropped": n_dims - n_kept,
            "n_runs": len(aurocs),
        }
        print(f"  {pct:>3d}% dims dropped ({n_dims - n_kept:>3d}/{n_dims} zeroed): "
              f"AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}  "
              f"[{len(aurocs)} seeds]")

    return results


# ===================================================================
# Experiment 3c: Label Noise Robustness
# ===================================================================
def exp_label_noise(adata):
    """
    Flip a fraction of eccrine<->hair labels in the TRAINING split and
    measure RF AUROC on CORRECT (unflipped) test labels. Tests whether
    the embedding space is robust to noisy supervision.
    """
    print("\n" + "=" * 60)
    print("Experiment 3c: Label Noise Robustness")
    print("=" * 60)

    X_all, y_all, _ = get_labeled_data(adata)
    noise_fracs = [0.0, 0.05, 0.10, 0.20, 0.50]
    results = {}

    for noise_frac in noise_fracs:
        aurocs = []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
            train_idx, test_idx = next(sss.split(X_all, y_all))

            X_train, y_train = X_all[train_idx], y_all[train_idx].copy()
            X_test, y_test = X_all[test_idx], y_all[test_idx]  # correct labels

            # Flip labels in training set
            if noise_frac > 0:
                n_flip = max(1, int(len(y_train) * noise_frac))
                flip_idx = rng.choice(len(y_train), size=n_flip, replace=False)
                y_train[flip_idx] = 1 - y_train[flip_idx]  # eccrine(1)<->hair(0)

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            auc = rf_auroc(X_train, y_train, X_test, y_test, seed=seed)
            aurocs.append(auc)

        pct = int(noise_frac * 100)
        results[pct] = {
            "mean": np.mean(aurocs),
            "std": np.std(aurocs),
            "n_flipped": int(len(train_idx) * noise_frac),
            "n_runs": len(aurocs),
        }
        print(f"  {pct:>3d}% labels flipped ({results[pct]['n_flipped']:>3d} cells): "
              f"AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}  "
              f"[{len(aurocs)} seeds]")

    return results


# ===================================================================
# Experiment 3d: Hyperparameter Documentation
# ===================================================================
def exp_hyperparameters():
    """Load and report the training configuration used."""
    print("\n" + "=" * 60)
    print("Experiment 3d: Hyperparameter Configuration")
    print("=" * 60)

    config = load_config()
    # Flatten for reporting
    flat = {}
    for section_name, section in config.items():
        if isinstance(section, dict):
            for k, v in section.items():
                flat[f"{section_name}.{k}"] = v
        else:
            flat[section_name] = section

    # Print key hyperparameters
    key_params = [
        ("encoder.n_layers", "Transformer layers"),
        ("encoder.n_heads", "Attention heads"),
        ("encoder.d_model", "Model dimension"),
        ("encoder.d_ff", "FFN dimension"),
        ("encoder.d_output", "Output dimension"),
        ("encoder.dropout", "Dropout"),
        ("encoder.lora_rank", "LoRA rank"),
        ("encoder.lora_alpha", "LoRA alpha"),
        ("contrastive.temperature_init", "Temperature (init)"),
        ("contrastive.alpha_max", "Hard-negative alpha_max"),
        ("contrastive.curriculum_warmup_epochs", "Curriculum warmup epochs"),
        ("contrastive.projection_dims", "Projection dimensions"),
        ("training.optimizer", "Optimizer"),
        ("training.lr_lora", "LR (LoRA)"),
        ("training.lr_head", "LR (head)"),
        ("training.weight_decay", "Weight decay"),
        ("training.batch_size", "Batch size"),
        ("training.n_epochs", "Max epochs"),
        ("training.early_stopping_patience", "Early stopping patience"),
        ("training.info_reg_lambda", "Info-reg lambda"),
        ("training.recon_weight", "Reconstruction weight"),
        ("training.scheduler", "LR scheduler"),
        ("training.gradient_clip", "Gradient clip"),
        ("data.n_hvgs", "Number of HVGs"),
        ("experiment.seed", "Random seed"),
    ]

    config_table = []
    for key, label in key_params:
        val = flat.get(key, "N/A")
        config_table.append((label, key, val))
        print(f"  {label:35s}: {val}")

    return config, config_table


# ===================================================================
# Results Formatting & Writing
# ===================================================================
def format_results_md(ds_results, dd_results, ln_results, config_table, elapsed):
    """Format all results into a single markdown block."""

    lines = []
    lines.append("**Sensitivity Analysis (Experiment 3)**\n")
    lines.append(f"Methodology: All tests use the EXISTING X_prism embeddings (128-d, "
                 f"from the trained PRISM-Encode model). A Random Forest classifier "
                 f"(100 trees) is trained on labeled cells (258 eccrine + 928 hair) "
                 f"with a 70/30 stratified train/test split. Each condition is "
                 f"repeated over {N_SEEDS} random seeds; we report mean +/- std RF AUROC.\n")

    # --- 3a: Downsampling ---
    lines.append("#### 3a. Downsampling Robustness\n")
    lines.append("How does RF AUROC degrade when the classifier sees fewer training cells?\n")
    lines.append("| Training Fraction | # Train Cells | RF AUROC (mean +/- std) |")
    lines.append("|:-----------------:|:------------:|:----------------------:|")
    for pct in sorted(ds_results.keys(), reverse=True):
        r = ds_results[pct]
        lines.append(f"| {pct}% | {r['n_train']} | {r['mean']:.4f} +/- {r['std']:.4f} |")

    # --- 3b: Dimension Dropout ---
    lines.append("\n#### 3b. Embedding Dimension Dropout Robustness\n")
    lines.append("How does RF AUROC degrade when embedding dimensions are randomly zeroed?\n")
    lines.append("| Dims Dropped | Dims Kept / 128 | RF AUROC (mean +/- std) |")
    lines.append("|:-----------:|:---------------:|:----------------------:|")
    for pct in sorted(dd_results.keys()):
        r = dd_results[pct]
        lines.append(f"| {pct}% | {r['n_dims_kept']}/128 | {r['mean']:.4f} +/- {r['std']:.4f} |")

    # --- 3c: Label Noise ---
    lines.append("\n#### 3c. Label Noise Robustness\n")
    lines.append("How does RF AUROC (on CORRECT test labels) degrade when training "
                 "labels have eccrine<->hair flips?\n")
    lines.append("| Noise Fraction | # Flipped | RF AUROC (mean +/- std) |")
    lines.append("|:-------------:|:---------:|:----------------------:|")
    for pct in sorted(ln_results.keys()):
        r = ln_results[pct]
        lines.append(f"| {pct}% | {r['n_flipped']} | {r['mean']:.4f} +/- {r['std']:.4f} |")

    # --- 3d: Hyperparameters ---
    lines.append("\n#### 3d. Hyperparameter Configuration\n")
    lines.append("Training configuration used to produce the PRISM embeddings "
                 "(from `configs/default.yaml`).\n")
    lines.append("| Parameter | Config Key | Value |")
    lines.append("|:----------|:-----------|:------|")
    for label, key, val in config_table:
        lines.append(f"| {label} | `{key}` | {val} |")

    lines.append(f"\nTotal experiment runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")

    return "\n".join(lines)


def main():
    total_start = time.time()

    print("=" * 60)
    print("PRISM Experiment 3: Sensitivity Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    adata = ad.read_h5ad(DATA_PATH)
    print(f"  Cells: {adata.shape[0]}, Genes: {adata.shape[1]}")
    print(f"  X_prism shape: {adata.obsm['X_prism'].shape}")

    X_labeled, y_labeled, mask = get_labeled_data(adata)
    n_eccrine = (y_labeled == 1).sum()
    n_hair = (y_labeled == 0).sum()
    print(f"  Labeled cells: {len(y_labeled)} (eccrine={n_eccrine}, hair={n_hair})")

    # Run experiments
    ds_results = exp_downsampling(adata)
    dd_results = exp_dimension_dropout(adata)
    ln_results = exp_label_noise(adata)
    config, config_table = exp_hyperparameters()

    elapsed = time.time() - total_start

    # Format results
    md_content = format_results_md(ds_results, dd_results, ln_results, config_table, elapsed)

    # Print to console
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(md_content)

    # Append to results.md
    with open(RESULTS_PATH, "a") as f:
        f.write(f"\n\n---\n\n### Experiment 3: Sensitivity Analysis\n")
        f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(md_content)
        f.write("\n")

    print(f"\nResults appended to {RESULTS_PATH}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
