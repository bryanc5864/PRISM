#!/usr/bin/env python
"""Regenerate per-system UMAP comparison figures with proper fate name labels."""

import os
import sys
import yaml
import numpy as np
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.utils.visualization import plot_umap_comparison

# Map system name -> (config path, adata path, figures dir)
SYSTEMS = {
    "skin": ("configs/skin.yaml", "data/processed/adata_processed.h5ad", "figures"),
    "pancreas": ("configs/pancreas.yaml", "data/processed/pancreas/adata_processed.h5ad", "figures/pancreas"),
    "cortex": ("configs/cortex.yaml", "data/processed/cortex/adata_processed.h5ad", "figures/cortex"),
    "hsc": ("configs/hsc.yaml", "data/processed/hsc/adata_processed.h5ad", "figures/hsc"),
    "cardiac": ("configs/cardiac.yaml", "data/processed/cardiac/adata_processed.h5ad", "figures/cardiac"),
    "intestine": ("configs/intestine.yaml", "data/processed/intestine/adata_processed.h5ad", "figures/intestine"),
    "lung": ("configs/lung.yaml", "data/processed/lung/adata_processed.h5ad", "figures/lung"),
    "neural_crest": ("configs/neural_crest.yaml", "data/processed/neural_crest/adata_processed.h5ad", "figures/neural_crest"),
    "oligo": ("configs/oligo.yaml", "data/processed/oligo/adata_processed.h5ad", "figures/oligo"),
    "thcell": ("configs/thcell.yaml", "data/processed/thcell/adata_processed.h5ad", "figures/thcell"),
    "paul": ("configs/paul.yaml", "data/processed/paul/adata_processed.h5ad", "figures/paul"),
    "sadefeldman": ("configs/sadefeldman.yaml", "data/processed/sadefeldman/adata_processed.h5ad", "figures/sadefeldman"),
}

for system, (config_path, adata_path, figures_dir) in SYSTEMS.items():
    if not os.path.exists(adata_path):
        print(f"[SKIP] {system}: {adata_path} not found")
        continue
    if not os.path.exists(config_path):
        print(f"[SKIP] {system}: {config_path} not found")
        continue

    print(f"\n{'='*60}")
    print(f"Regenerating UMAP for {system}")
    print(f"{'='*60}")

    # Load config to get fate_categories
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    fate_categories = cfg["system"]["fate_categories"]
    label_names = {i: name for i, name in enumerate(fate_categories)}
    print(f"  Label names: {label_names}")

    # Load adata
    print(f"  Loading {adata_path} ...")
    adata = ad.read_h5ad(adata_path)

    # Get integer fate labels
    if "fate_int" in adata.obs.columns:
        labels = adata.obs["fate_int"].values.astype(int)
    elif "fate_label" in adata.obs.columns:
        # Try to map string labels to ints via fate_categories
        fl = adata.obs["fate_label"].values
        cat_to_int = {cat: i for i, cat in enumerate(fate_categories)}
        labels = np.array([cat_to_int.get(str(x), 0) for x in fl])
    else:
        print(f"  [SKIP] No fate_label or fate_int column")
        continue

    print(f"  Labels: unique={np.unique(labels)}, n_cells={len(labels)}")

    # Build embeddings dict
    embeddings_dict = {}
    if "X_pca" in adata.obsm:
        pca = np.array(adata.obsm["X_pca"])
        embeddings_dict["PCA"] = pca[:, :min(30, pca.shape[1])]
    if "X_harmony" in adata.obsm:
        harm = np.array(adata.obsm["X_harmony"])
        embeddings_dict["Harmony"] = harm[:, :min(30, harm.shape[1])]
    if "X_prism" in adata.obsm:
        embeddings_dict["PRISM"] = np.array(adata.obsm["X_prism"])

    if not embeddings_dict:
        print(f"  [SKIP] No embeddings found")
        continue

    print(f"  Methods: {list(embeddings_dict.keys())}")

    os.makedirs(figures_dir, exist_ok=True)
    save_path = f"{figures_dir}/umap_comparison.png"

    plot_umap_comparison(
        embeddings_dict,
        labels,
        label_names=label_names,
        save_path=save_path,
    )
    print(f"  Saved {save_path}")

print("\nDone!")
