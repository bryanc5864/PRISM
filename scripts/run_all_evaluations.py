#!/usr/bin/env python
"""Comprehensive evaluation: all baselines x all metrics x all 15 systems.

Runs on existing embeddings (X_prism, X_pca, X_harmony) plus computes
DiffMap, UMAP, XGBoost, scGPT, Geneformer embeddings where available.
Computes full extended metrics suite for every method on every system.

Output: data/processed/full_evaluation_results.json
        data/processed/full_evaluation_results.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import anndata as ad
import scanpy as sc
from collections import OrderedDict

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.utils.metrics import compute_extended_metrics


# ---------------------------------------------------------------------------
# System definitions
# ---------------------------------------------------------------------------
SYSTEMS = OrderedDict([
    ("skin", {
        "adata": "data/processed/adata_processed.h5ad",
        "config": "configs/skin.yaml",
        "scgpt": "data/processed/scgpt_embeddings.npy",
        "geneformer": "data/processed/geneformer_embeddings.npy",
    }),
    ("pancreas", {
        "adata": "data/processed/pancreas/adata_processed.h5ad",
        "config": "configs/pancreas.yaml",
        "scgpt": "data/processed/pancreas/scgpt_embeddings.npy",
        "geneformer": "data/processed/pancreas/geneformer_embeddings.npy",
    }),
    ("cortex", {
        "adata": "data/processed/cortex/adata_processed.h5ad",
        "config": "configs/cortex.yaml",
        "scgpt": "data/processed/cortex/scgpt_embeddings.npy",
        "geneformer": "data/processed/cortex/geneformer_embeddings.npy",
    }),
    ("hsc", {
        "adata": "data/processed/hsc/adata_processed.h5ad",
        "config": "configs/hsc.yaml",
        "scgpt": "data/processed/hsc/scgpt_embeddings.npy",
        "geneformer": "data/processed/hsc/geneformer_embeddings.npy",
    }),
    ("cardiac", {
        "adata": "data/processed/cardiac/adata_processed.h5ad",
        "config": "configs/cardiac.yaml",
        "scgpt": "data/processed/cardiac/scgpt_embeddings.npy",
        "geneformer": "data/processed/cardiac/geneformer_embeddings.npy",
    }),
    ("intestine", {
        "adata": "data/processed/intestine/adata_processed.h5ad",
        "config": "configs/intestine.yaml",
        "scgpt": "data/processed/intestine/scgpt_embeddings.npy",
        "geneformer": "data/processed/intestine/geneformer_embeddings.npy",
    }),
    ("lung", {
        "adata": "data/processed/lung/adata_processed.h5ad",
        "config": "configs/lung.yaml",
        "scgpt": "data/processed/lung/scgpt_embeddings.npy",
        "geneformer": "data/processed/lung/geneformer_embeddings.npy",
    }),
    ("neural_crest", {
        "adata": "data/processed/neural_crest/adata_processed.h5ad",
        "config": "configs/neural_crest.yaml",
        "scgpt": "data/processed/neural_crest/scgpt_embeddings.npy",
        "geneformer": "data/processed/neural_crest/geneformer_embeddings.npy",
    }),
    ("oligo", {
        "adata": "data/processed/oligo/adata_processed.h5ad",
        "config": "configs/oligo.yaml",
        "scgpt": "data/processed/oligo/scgpt_embeddings.npy",
        "geneformer": "data/processed/oligo/geneformer_embeddings.npy",
    }),
    ("thcell", {
        "adata": "data/processed/thcell/adata_processed.h5ad",
        "config": "configs/thcell.yaml",
        "scgpt": "data/processed/thcell/scgpt_embeddings.npy",
        "geneformer": "data/processed/thcell/geneformer_embeddings.npy",
    }),
    ("paul", {
        "adata": "data/processed/paul/adata_processed.h5ad",
        "config": "configs/paul.yaml",
        "scgpt": "data/processed/paul/scgpt_embeddings.npy",
        "geneformer": "data/processed/paul/geneformer_embeddings.npy",
    }),
    ("nestorowa", {
        "adata": "data/processed/nestorowa/adata_processed.h5ad",
        "config": "configs/nestorowa.yaml",
        "scgpt": "data/processed/nestorowa/scgpt_embeddings.npy",
        "geneformer": "data/processed/nestorowa/geneformer_embeddings.npy",
    }),
    ("sadefeldman", {
        "adata": "data/processed/sadefeldman/adata_processed.h5ad",
        "config": "configs/sadefeldman.yaml",
        "scgpt": "data/processed/sadefeldman/scgpt_embeddings.npy",
        "geneformer": "data/processed/sadefeldman/geneformer_embeddings.npy",
    }),
    ("tirosh_melanoma", {
        "adata": "data/processed/tirosh_melanoma/adata_processed.h5ad",
        "config": "configs/tirosh_melanoma.yaml",
        "scgpt": "data/processed/tirosh_melanoma/scgpt_embeddings.npy",
        "geneformer": "data/processed/tirosh_melanoma/geneformer_embeddings.npy",
    }),
    ("neftel_gbm", {
        "adata": "data/processed/neftel_gbm/adata_processed.h5ad",
        "config": "configs/neftel_gbm.yaml",
        "scgpt": "data/processed/neftel_gbm/scgpt_embeddings.npy",
        "geneformer": "data/processed/neftel_gbm/geneformer_embeddings.npy",
    }),
])


def get_condition_key(config_path):
    """Read condition_key from system YAML config."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["system"].get("condition_key", "genotype")


def compute_diffmap_embedding(adata):
    """Compute diffusion map embedding."""
    try:
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        adata_copy = adata.copy()
        sc.pp.neighbors(adata_copy, use_rep="X_pca", n_neighbors=30)
        sc.tl.diffmap(adata_copy, n_comps=15)
        return np.array(adata_copy.obsm["X_diffmap"])
    except Exception as e:
        print(f"    DiffMap failed: {e}")
        return None


def compute_umap_embedding(adata):
    """Compute UMAP embedding for classification."""
    try:
        from umap import UMAP
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        reducer = UMAP(n_components=10, random_state=42, n_neighbors=30, min_dist=0.1)
        return reducer.fit_transform(adata.obsm["X_pca"][:, :50])
    except Exception as e:
        print(f"    UMAP embedding failed: {e}")
        return None


def compute_harmony_embedding(adata, condition_key):
    """Compute Harmony embedding if not already present."""
    if "X_harmony" in adata.obsm:
        return np.array(adata.obsm["X_harmony"])
    try:
        import harmonypy as hm
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        batch_key = condition_key
        if batch_key not in adata.obs.columns or adata.obs[batch_key].nunique() < 2:
            for key in ["sample", "batch", "library", "plate", "condition", "stage", "timepoint", "cell_type"]:
                if key in adata.obs.columns and adata.obs[key].nunique() > 1:
                    batch_key = key
                    break
        ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key, max_iter_harmony=20)
        return ho.Z_corr
    except Exception as e:
        print(f"    Harmony failed: {e}")
        return None


def get_batch_labels(adata, condition_key):
    """Get batch labels for iLISI/batch entropy computation."""
    for key in [condition_key, "sample", "batch", "library", "plate"]:
        if key in adata.obs.columns and adata.obs[key].nunique() > 1:
            return adata.obs[key].astype(str).values
    return None


def evaluate_system(system_name, system_info):
    """Run all baselines and extended metrics for one system."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {system_name}")
    print(f"{'='*70}")

    adata_path = system_info["adata"]
    if not os.path.exists(adata_path):
        print(f"  SKIP: {adata_path} not found")
        return None

    adata = ad.read_h5ad(adata_path)
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Get labels
    if "fate_int" in adata.obs.columns:
        labels = adata.obs["fate_int"].values.astype(int)
    elif "fate_label" in adata.obs.columns:
        # Encode string labels as ints
        cats = adata.obs["fate_label"].astype("category").cat.categories
        labels = adata.obs["fate_label"].astype("category").cat.codes.values
    else:
        print(f"  SKIP: no fate labels")
        return None

    print(f"  Labels: {np.unique(labels)}, distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Get batch labels and original expression
    condition_key = get_condition_key(system_info["config"])
    batch_labels = get_batch_labels(adata, condition_key)
    X_original = None
    if hasattr(adata.X, "toarray"):
        if adata.n_obs * adata.n_vars < 500_000_000:  # < 500M entries
            X_original = adata.X.toarray()
    elif isinstance(adata.X, np.ndarray):
        X_original = adata.X

    # Collect all embeddings
    embeddings = OrderedDict()

    # --- PRISM ---
    if "X_prism" in adata.obsm:
        embeddings["PRISM"] = np.array(adata.obsm["X_prism"])
        print(f"  PRISM: {embeddings['PRISM'].shape}")

    # --- PCA ---
    if "X_pca" in adata.obsm:
        pca = np.array(adata.obsm["X_pca"])
        embeddings["PCA"] = pca[:, :min(50, pca.shape[1])]
        print(f"  PCA: {embeddings['PCA'].shape}")
    else:
        sc.pp.pca(adata, n_comps=50)
        embeddings["PCA"] = np.array(adata.obsm["X_pca"])
        print(f"  PCA: computed {embeddings['PCA'].shape}")

    # --- Harmony ---
    harmony_emb = compute_harmony_embedding(adata, condition_key)
    if harmony_emb is not None:
        embeddings["Harmony"] = harmony_emb[:, :min(50, harmony_emb.shape[1])]
        print(f"  Harmony: {embeddings['Harmony'].shape}")

    # --- DiffMap ---
    diffmap_emb = compute_diffmap_embedding(adata)
    if diffmap_emb is not None:
        embeddings["DiffMap"] = diffmap_emb
        print(f"  DiffMap: {embeddings['DiffMap'].shape}")

    # --- UMAP (10D for classification) ---
    umap_emb = compute_umap_embedding(adata)
    if umap_emb is not None:
        embeddings["UMAP"] = umap_emb
        print(f"  UMAP: {embeddings['UMAP'].shape}")

    # --- scGPT ---
    if os.path.exists(system_info.get("scgpt", "")):
        scgpt = np.load(system_info["scgpt"])
        if scgpt.shape[0] == adata.n_obs:
            embeddings["scGPT"] = scgpt
            print(f"  scGPT: {scgpt.shape}")
        else:
            print(f"  scGPT: shape mismatch ({scgpt.shape[0]} vs {adata.n_obs})")

    # --- Geneformer ---
    if os.path.exists(system_info.get("geneformer", "")):
        gf = np.load(system_info["geneformer"])
        if gf.shape[0] == adata.n_obs:
            embeddings["Geneformer"] = gf
            print(f"  Geneformer: {gf.shape}")
        else:
            print(f"  Geneformer: shape mismatch ({gf.shape[0]} vs {adata.n_obs})")

    # --- Evaluate all methods ---
    results = {}
    for method_name, emb in embeddings.items():
        print(f"\n  Computing metrics for {method_name} ...")

        # Handle NaN/Inf
        if np.any(~np.isfinite(emb)):
            print(f"    WARNING: {method_name} has non-finite values, replacing with 0")
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            metrics = compute_extended_metrics(
                emb, labels,
                batch_labels=batch_labels,
                X_original=X_original,
                method_name=method_name,
            )
            results[method_name] = metrics

            # Print key metrics
            ari = metrics.get("ARI", "N/A")
            auroc = metrics.get("RF_AUROC", "N/A")
            asw = metrics.get("ASW", "N/A")
            knn = metrics.get("kNN_purity@10", "N/A")
            brier = metrics.get("RF_Brier", "N/A")
            if isinstance(ari, float):
                print(f"    ARI={ari:.3f}  RF_AUROC={auroc:.3f}  ASW={asw:.3f}  kNN@10={knn:.3f}  Brier={brier:.3f}")
            else:
                print(f"    {metrics}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[method_name] = {"method": method_name, "error": str(e)}

    # Save per-system results
    out_dir = os.path.dirname(adata_path)
    out_path = os.path.join(out_dir, "full_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    return results


def results_to_csv(all_results, csv_path):
    """Convert nested results dict to a flat CSV."""
    import csv

    # Collect all metric keys
    all_metrics = set()
    for sys_results in all_results.values():
        if sys_results is None:
            continue
        for method_results in sys_results.values():
            all_metrics.update(method_results.keys())
    all_metrics.discard("method")
    all_metrics.discard("error")
    metric_cols = sorted(all_metrics)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "method"] + metric_cols)
        for sys_name, sys_results in all_results.items():
            if sys_results is None:
                continue
            for method_name, metrics in sys_results.items():
                row = [sys_name, method_name]
                for col in metric_cols:
                    val = metrics.get(col, "")
                    if isinstance(val, float):
                        row.append(f"{val:.4f}")
                    else:
                        row.append(str(val) if val != "" else "")
                writer.writerow(row)

    print(f"\nSaved CSV: {csv_path}")


def main():
    print("=" * 70)
    print("  Comprehensive PRISM Evaluation")
    print(f"  Systems: {len(SYSTEMS)}")
    print("  Methods: PRISM, PCA, Harmony, DiffMap, UMAP, scGPT, Geneformer")
    print("  Metrics: ARI, AMI, NMI, ASW, Cohen's kappa, RF_AUROC, RF_F1,")
    print("           LR_AUROC, kNN purity, cLISI, iLISI, batch entropy,")
    print("           ECE, Brier, trustworthiness")
    print("=" * 70)

    all_results = OrderedDict()

    for sys_name, sys_info in SYSTEMS.items():
        results = evaluate_system(sys_name, sys_info)
        all_results[sys_name] = results

    # Save combined results
    combined_path = "data/processed/full_evaluation_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved combined JSON: {combined_path}")

    # Save CSV
    csv_path = "data/processed/full_evaluation_results.csv"
    results_to_csv(all_results, csv_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: RF_AUROC by System x Method")
    print("=" * 70)
    methods_seen = set()
    for sys_results in all_results.values():
        if sys_results:
            methods_seen.update(sys_results.keys())
    methods_order = ["PRISM", "PCA", "Harmony", "DiffMap", "UMAP", "scGPT", "Geneformer"]
    methods_order = [m for m in methods_order if m in methods_seen]

    header = f"{'System':<18s}" + "".join(f"{m:<12s}" for m in methods_order)
    print(header)
    print("-" * len(header))
    for sys_name, sys_results in all_results.items():
        if sys_results is None:
            print(f"{sys_name:<18s} (no data)")
            continue
        row = f"{sys_name:<18s}"
        for m in methods_order:
            val = sys_results.get(m, {}).get("RF_AUROC", "")
            if isinstance(val, float):
                row += f"{val:<12.3f}"
            else:
                row += f"{'---':<12s}"
        print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
