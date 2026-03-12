#!/usr/bin/env python3
"""
Re-process HSC data with clone matrix and run clonal validation.

Strategy:
1. Delete raw cache → re-load from extracted files (picks up clone_matrix)
2. Re-preprocess with annotation-based labels
3. Transfer X_prism from existing checkpoint
4. Run clonal validation
"""

import os
import sys
import numpy as np

# Fix library paths
import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import anndata as ad
from prism.config import SystemConfig
from prism.data.download_hsc import download_hsc
from prism.data.preprocess import preprocess_adata, assign_genotypes, assign_labels, compute_harmony_baseline
from prism.experiments.clonal_validation import run_clonal_validation

# Load system config
system_config = SystemConfig.from_yaml("configs/hsc.yaml")
processed_dir = "data/processed/hsc"
figures_dir = "figures/hsc"
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Step 1: Load existing processed data (has X_prism)
print("=" * 60)
print("Step 1: Load existing processed HSC data")
print("=" * 60)
old_adata = ad.read_h5ad(os.path.join(processed_dir, "adata_processed.h5ad"))
print(f"  Existing data: {old_adata.shape}")
print(f"  Has X_prism: {'X_prism' in old_adata.obsm}")
old_prism = old_adata.obsm["X_prism"].copy() if "X_prism" in old_adata.obsm else None
old_obs_names = old_adata.obs_names.tolist()

# Step 2: Delete raw cache and re-download (to get clone_matrix)
print("\n" + "=" * 60)
print("Step 2: Re-load raw data with clone matrix")
print("=" * 60)
cache_path = "data/raw/hsc_combined.h5ad"
if os.path.exists(cache_path):
    os.remove(cache_path)
    print(f"  Deleted cache: {cache_path}")

raw_adata = download_hsc(raw_dir="data/raw", force=False)
print(f"  Raw data: {raw_adata.shape}")
print(f"  Has clone_matrix: {'clone_matrix' in raw_adata.obsm}")
print(f"  Obs columns: {list(raw_adata.obs.columns[:15])}")

# Step 3: Preprocess with annotation-based labels
print("\n" + "=" * 60)
print("Step 3: Preprocess with annotation labels")
print("=" * 60)
adata = preprocess_adata(
    raw_adata,
    min_genes=200,
    max_genes=5000,
    max_mito_pct=5.0,
    n_hvgs=2000,
    forced_genes=system_config.forced_genes,
)

adata = assign_genotypes(
    adata,
    sample_condition_map=system_config.sample_condition_map or None,
    condition_key=system_config.condition_key,
)

adata = assign_labels(
    adata,
    condition_key=system_config.condition_key,
    conditions=system_config.conditions,
    marker_scores=system_config.marker_scores or None,
    fate_categories=system_config.fate_categories,
    label_strategy=system_config.label_strategy,
    annotation_key=system_config.annotation_key,
    annotation_fate_map=system_config.annotation_fate_map or None,
)

adata = compute_harmony_baseline(adata)

# Step 4: Transfer X_prism from old processed data
print("\n" + "=" * 60)
print("Step 4: Transfer PRISM embeddings")
print("=" * 60)
if old_prism is not None:
    # Match cells by obs_names
    old_name_to_idx = {name: i for i, name in enumerate(old_obs_names)}
    new_names = adata.obs_names.tolist()

    matched = 0
    prism_emb = np.zeros((adata.shape[0], old_prism.shape[1]), dtype=np.float32)
    for i, name in enumerate(new_names):
        if name in old_name_to_idx:
            prism_emb[i] = old_prism[old_name_to_idx[name]]
            matched += 1

    print(f"  Matched {matched}/{adata.shape[0]} cells from old embeddings")

    if matched > adata.shape[0] * 0.5:
        adata.obsm["X_prism"] = prism_emb
        print(f"  Transferred X_prism: {prism_emb.shape}")
    else:
        print("  WARNING: Too few matched cells. Using X_pca as fallback for validation.")
else:
    print("  No old X_prism found. Will use X_pca for validation.")

# Step 5: Save processed data
print("\n" + "=" * 60)
print("Step 5: Save processed data")
print("=" * 60)
save_path = os.path.join(processed_dir, "adata_processed.h5ad")
adata.write_h5ad(save_path)
print(f"  Saved to {save_path}")
print(f"  Shape: {adata.shape}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  Has clone_matrix: {'clone_matrix' in adata.obsm}")
print(f"  Has time_point: {'time_point' in adata.obs.columns}")

# Step 6: Run clonal validation
print("\n" + "=" * 60)
print("Step 6: Clonal Validation")
print("=" * 60)
embedding_key = "X_prism" if "X_prism" in adata.obsm else "X_pca"
print(f"  Using embedding: {embedding_key}")

results = run_clonal_validation(
    adata,
    fate_col="fate_label",
    time_col="time_point",
    embedding_key=embedding_key,
    save_dir=figures_dir,
)

print("\n" + "=" * 60)
print("DONE: HSC Clonal Validation Complete")
print("=" * 60)

# Print summary
conc = results.get("concordance", {})
if "error" not in conc:
    print(f"  Concordance rate: {conc['concordance_rate']:.3f} ({conc['n_tested_clones']} clones)")

pur = results.get("purity", {})
if "error" not in pur:
    print(f"  Clonal purity: {pur['mean_purity']:.3f} (random baseline: {pur['random_baseline']:.3f})")

pred = results.get("predictability", {})
if "error" not in pred:
    print(f"  Fate predictability AUROC: {pred['auroc']:.3f} (accuracy: {pred['accuracy']:.3f})")
