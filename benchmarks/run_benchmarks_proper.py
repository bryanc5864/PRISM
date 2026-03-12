#!/usr/bin/env python3
"""
Proper benchmarking of PRISM vs standard methods.

Demonstrates Section 2.2: "When Cells Are Too Similar for Current Tools"

Tests:
1. Seurat/Scanpy PCA pipeline (varying # PCs: 10, 30, 50, all)
2. PCA variance analysis — where do discriminatory genes fall?
3. Harmony batch correction
4. scVI variational autoencoder
5. PRISM-Encode

For each method, evaluates:
- Clustering metrics (ARI, AMI, ASW)
- Classification (RF/LR F1, AUROC)
- Whether eccrine vs hair progenitors are separable
"""

import os
import sys
import time
import numpy as np
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

warnings.filterwarnings("ignore")

import anndata as ad
import scanpy as sc
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
    silhouette_score, f1_score, roc_auc_score
)
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize
import scipy.sparse as sp


def compute_metrics(embeddings, labels, method_name=""):
    """Compute full suite of clustering + classification metrics."""
    metrics = {"method": method_name}

    # Filter to cells with known fate labels (eccrine=2, hair=3)
    # For clustering metrics, use all labeled cells
    valid = labels > 0  # exclude non_appendage (0) for fair comparison
    emb_valid = embeddings[valid]
    lab_valid = labels[valid]

    if len(np.unique(lab_valid)) < 2:
        return {**metrics, "error": "too_few_classes"}

    # Clustering metrics
    kmeans = KMeans(n_clusters=min(4, len(np.unique(lab_valid))), random_state=42, n_init=10)
    pred = kmeans.fit_predict(emb_valid)
    metrics["ARI"] = adjusted_rand_score(lab_valid, pred)
    metrics["AMI"] = adjusted_mutual_info_score(lab_valid, pred)
    metrics["NMI"] = normalized_mutual_info_score(lab_valid, pred)

    try:
        metrics["ASW"] = silhouette_score(emb_valid, lab_valid, sample_size=min(5000, len(lab_valid)))
    except Exception:
        metrics["ASW"] = 0.0

    # Binary classification: eccrine (2) vs hair (3) — the key test
    fate_mask = (labels == 2) | (labels == 3)
    if fate_mask.sum() >= 20:
        emb_fate = embeddings[fate_mask]
        lab_fate = (labels[fate_mask] == 2).astype(int)  # 1=eccrine, 0=hair

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_pred = cross_val_predict(rf, emb_fate, lab_fate, cv=5)
        rf_prob = cross_val_predict(rf, emb_fate, lab_fate, cv=5, method="predict_proba")[:, 1]
        metrics["RF_F1_eccrine_vs_hair"] = f1_score(lab_fate, rf_pred, average="binary")
        metrics["RF_AUROC_eccrine_vs_hair"] = roc_auc_score(lab_fate, rf_prob)

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr_pred = cross_val_predict(lr, emb_fate, lab_fate, cv=5)
        lr_prob = cross_val_predict(lr, emb_fate, lab_fate, cv=5, method="predict_proba")[:, 1]
        metrics["LR_F1_eccrine_vs_hair"] = f1_score(lab_fate, lr_pred, average="binary")
        metrics["LR_AUROC_eccrine_vs_hair"] = roc_auc_score(lab_fate, lr_prob)

        metrics["n_eccrine"] = int((lab_fate == 1).sum())
        metrics["n_hair"] = int((lab_fate == 0).sum())
    else:
        metrics["RF_F1_eccrine_vs_hair"] = 0.0
        metrics["RF_AUROC_eccrine_vs_hair"] = 0.5
        metrics["LR_F1_eccrine_vs_hair"] = 0.0
        metrics["LR_AUROC_eccrine_vs_hair"] = 0.5

    # Full multi-class classification (all labels)
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_full_pred = cross_val_predict(rf_full, embeddings, labels, cv=5)
    metrics["RF_F1_macro"] = f1_score(labels, rf_full_pred, average="macro")

    return metrics


def analyze_pca_variance(adata, labels):
    """Analyze where discriminatory signal falls in PCA space.

    Key prediction from Section 2.2: eccrine-specific genes should have
    negligible loading on top PCs because >90% of variance is shared.
    """
    print("\n" + "="*60)
    print("PCA VARIANCE ANALYSIS: Where Does the Discriminatory Signal Live?")
    print("="*60)

    # Known eccrine/hair markers
    eccrine_markers = ["En1", "Wnt10a", "Wnt10b", "Edar", "Eda", "Dkk4", "Lgr6", "Tfap2b"]
    hair_markers = ["Shh", "Bmp4", "Bmp2", "Sox9", "Lhx2", "Wnt5a", "Foxd1", "Ptch1"]

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)

    # Get PCA loadings
    if hasattr(adata.uns.get("pca", {}), "get"):
        loadings = adata.varm.get("PCs", None)
    else:
        loadings = None

    if loadings is not None:
        var_ratio = adata.uns["pca"]["variance_ratio"]
        print(f"\nVariance explained by top PCs:")
        print(f"  PC1-10:  {var_ratio[:10].sum():.1%}")
        print(f"  PC11-30: {var_ratio[10:30].sum():.1%}")
        print(f"  PC31-50: {var_ratio[30:50].sum():.1%}")

        # Where do marker genes load?
        gene_names = adata.var_names.tolist()
        print(f"\nMarker gene loadings (absolute value, top 5 PCs):")
        for marker_set, markers, label in [
            (eccrine_markers, eccrine_markers, "ECCRINE"),
            (hair_markers, hair_markers, "HAIR"),
        ]:
            print(f"\n  {label} markers:")
            for gene in markers:
                if gene in gene_names:
                    idx = gene_names.index(gene)
                    gene_loadings = np.abs(loadings[idx, :5])
                    max_pc = np.argmax(np.abs(loadings[idx, :50])) + 1
                    max_loading = np.abs(loadings[idx, :50]).max()
                    print(f"    {gene:10s}: PC1={gene_loadings[0]:.4f}, PC2={gene_loadings[1]:.4f}, "
                          f"PC3={gene_loadings[2]:.4f} | max: PC{max_pc} ({max_loading:.4f})")

    # Quantify: can top PCs separate eccrine vs hair?
    fate_mask = (labels == 2) | (labels == 3)
    if fate_mask.sum() > 0:
        print(f"\n{'='*60}")
        print("ECCRINE vs HAIR SEPARABILITY BY PC RANGE")
        print(f"  Eccrine cells: {(labels == 2).sum()}")
        print(f"  Hair cells: {(labels == 3).sum()}")
        print(f"{'='*60}")

        pca_emb = adata.obsm["X_pca"]
        for pc_range, label in [(slice(0, 5), "PC1-5"), (slice(0, 10), "PC1-10"),
                                 (slice(0, 30), "PC1-30"), (slice(0, 50), "PC1-50"),
                                 (slice(30, 50), "PC31-50")]:
            emb = pca_emb[fate_mask, pc_range]
            lab = (labels[fate_mask] == 2).astype(int)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            pred = cross_val_predict(rf, emb, lab, cv=5, method="predict_proba")[:, 1]
            auroc = roc_auc_score(lab, pred)
            print(f"  {label:10s}: AUROC = {auroc:.4f}")


def run_seurat_scanpy_pipeline(adata, labels, n_pcs_list=[10, 30, 50]):
    """Seurat/Scanpy standard PCA + clustering pipeline."""
    results = {}

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)

    for n_pcs in n_pcs_list:
        name = f"PCA(top{n_pcs})+Leiden"
        print(f"\n--- {name} ---")

        emb = adata.obsm["X_pca"][:, :n_pcs]

        # Leiden clustering (Seurat-style)
        sc.pp.neighbors(adata, n_pcs=n_pcs, use_rep="X_pca",
                        key_added=f"pca{n_pcs}_nn")
        sc.tl.leiden(adata, resolution=1.0,
                     key_added=f"leiden_pca{n_pcs}",
                     neighbors_key=f"pca{n_pcs}_nn")

        metrics = compute_metrics(emb, labels, method_name=name)
        results[name] = metrics
        print(f"  ARI={metrics['ARI']:.4f}, AUROC(ecc/hair)={metrics.get('RF_AUROC_eccrine_vs_hair', 'N/A')}")

    # Also test KMeans on PCA
    for n_pcs in [30, 50]:
        name = f"PCA(top{n_pcs})+KMeans"
        emb = adata.obsm["X_pca"][:, :n_pcs]
        metrics = compute_metrics(emb, labels, method_name=name)
        results[name] = metrics

    return results


def run_harmony(adata, labels):
    """Harmony batch correction + clustering."""
    try:
        import harmonypy as hm

        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)

        batch_key = "sample" if "sample" in adata.obs else "genotype"
        ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key, max_iter_harmony=20)
        emb = ho.Z_corr

        for n_pcs in [30, 50]:
            name = f"Harmony(top{n_pcs})+Leiden"
            sub_emb = emb[:, :n_pcs]
            metrics = compute_metrics(sub_emb, labels, method_name=name)
            print(f"\n--- {name} ---")
            print(f"  ARI={metrics['ARI']:.4f}, AUROC(ecc/hair)={metrics.get('RF_AUROC_eccrine_vs_hair', 'N/A')}")
            yield name, metrics

    except ImportError:
        print("harmonypy not installed")


def run_scvi_benchmark(adata, labels):
    """scVI with proper setup."""
    import site, os
    try:
        env = {**os.environ}
        sp = site.getsitepackages()[0]
        csp = os.path.join(sp, "nvidia", "cusparselt", "lib")
        if os.path.exists(csp):
            env["LD_LIBRARY_PATH"] = csp + ":" + env.get("LD_LIBRARY_PATH", "")

        # Save HVG data for subprocess
        hvg = adata.var["highly_variable"] if "highly_variable" in adata.var else np.ones(adata.shape[1], dtype=bool)
        adata_hvg = adata[:, hvg].copy()
        adata_hvg.write_h5ad("/tmp/bench_scvi.h5ad")
        del adata_hvg

        script = """
import anndata as ad, numpy as np, sys, warnings
warnings.filterwarnings("ignore")
try:
    import scvi
    adata = ad.read_h5ad("/tmp/bench_scvi.h5ad")
    scvi.model.SCVI.setup_anndata(adata, batch_key="sample" if "sample" in adata.obs else None)
    model = scvi.model.SCVI(adata, n_latent=30)
    model.train(max_epochs=200, early_stopping=True, progress_bar=False)
    emb = model.get_latent_representation()
    np.save("/tmp/bench_scvi_emb.npy", emb)
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {e}")
"""
        result = subprocess.run(["python3", "-c", script],
                                capture_output=True, text=True, timeout=600, env=env)

        if "SUCCESS" in result.stdout:
            emb = np.load("/tmp/bench_scvi_emb.npy")
            metrics = compute_metrics(emb, labels, method_name="scVI")
            print(f"\n--- scVI ---")
            print(f"  ARI={metrics['ARI']:.4f}, AUROC(ecc/hair)={metrics.get('RF_AUROC_eccrine_vs_hair', 'N/A')}")
            return {"scVI": metrics}
        else:
            print(f"scVI failed: {result.stdout[-300:]}")
            return {"scVI": {"method": "scVI", "error": "failed"}}
    except Exception as e:
        return {"scVI": {"method": "scVI", "error": str(e)}}


def run_prism(adata, labels):
    """Load PRISM embeddings from trained model."""
    if "X_prism" in adata.obsm:
        emb = adata.obsm["X_prism"]
        metrics = compute_metrics(emb, labels, method_name="PRISM")
        print(f"\n--- PRISM ---")
        print(f"  ARI={metrics['ARI']:.4f}, AUROC(ecc/hair)={metrics.get('RF_AUROC_eccrine_vs_hair', 'N/A')}")
        return {"PRISM": metrics}
    else:
        print("WARNING: X_prism not found in adata")
        return {}


def main():
    print("="*60)
    print("PRISM BENCHMARK: Standard Methods vs Cryptic Cell Fates")
    print("Section 2.2: When Cells Are Too Similar for Current Tools")
    print("="*60)

    # Load data
    adata = ad.read_h5ad(os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad"))
    print(f"\nDataset: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Get fate labels
    label_map = {"non_appendage": 0, "undetermined": 1, "eccrine": 2, "hair": 3}
    if "fate_label" in adata.obs:
        labels = adata.obs["fate_label"].map(label_map).fillna(0).astype(int).values
    elif "fate_int" in adata.obs:
        labels = adata.obs["fate_int"].values.astype(int)
    else:
        print("ERROR: No fate labels found")
        return

    eccrine_n = (labels == 2).sum()
    hair_n = (labels == 3).sum()
    print(f"Eccrine progenitors: {eccrine_n}")
    print(f"Hair progenitors: {hair_n}")
    print(f"Challenge: separate {eccrine_n} eccrine from {hair_n} hair cells")
    print(f"  where >90% of transcriptome is shared")

    all_results = {}

    # 1. PCA variance analysis
    analyze_pca_variance(adata, labels)

    # 2. Seurat/Scanpy PCA pipeline
    print("\n" + "="*60)
    print("BENCHMARK 1: Seurat/Scanpy PCA + Clustering")
    print("="*60)
    pca_results = run_seurat_scanpy_pipeline(adata, labels)
    all_results.update(pca_results)

    # 3. Harmony
    print("\n" + "="*60)
    print("BENCHMARK 2: Harmony Batch Correction")
    print("="*60)
    for name, metrics in run_harmony(adata, labels):
        all_results[name] = metrics

    # 4. scVI
    print("\n" + "="*60)
    print("BENCHMARK 3: scVI (Variational Autoencoder)")
    print("="*60)
    scvi_results = run_scvi_benchmark(adata, labels)
    all_results.update(scvi_results)

    # 5. PRISM
    print("\n" + "="*60)
    print("BENCHMARK 4: PRISM (Our Method)")
    print("="*60)
    prism_results = run_prism(adata, labels)
    all_results.update(prism_results)

    # Summary table
    print("\n" + "="*60)
    print("FINAL RESULTS: Eccrine vs Hair Progenitor Resolution")
    print("="*60)
    print(f"\n{'Method':<30} {'ARI':>6} {'AMI':>6} {'ASW':>6} {'RF_AUROC':>10} {'LR_AUROC':>10}")
    print("-"*75)
    for name, m in sorted(all_results.items(), key=lambda x: x[1].get("RF_AUROC_eccrine_vs_hair", 0)):
        if "error" in m:
            print(f"{name:<30} {'FAILED':>6}")
            continue
        print(f"{name:<30} {m.get('ARI', 0):>6.3f} {m.get('AMI', 0):>6.3f} "
              f"{m.get('ASW', 0):>6.3f} {m.get('RF_AUROC_eccrine_vs_hair', 0):>10.4f} "
              f"{m.get('LR_AUROC_eccrine_vs_hair', 0):>10.4f}")

    # Write to results.md
    result_text = "**Proper Benchmark: Standard Methods vs Cryptic Cell Fates**\n\n"
    result_text += f"Challenge: Separate {eccrine_n} eccrine from {hair_n} hair progenitors\n"
    result_text += "where >90% of the transcriptome is shared (Section 2.2)\n\n"
    result_text += "| Method | ARI | AMI | ASW | RF AUROC (ecc/hair) | LR AUROC (ecc/hair) |\n"
    result_text += "|--------|-----|-----|-----|---------------------|---------------------|\n"

    for name, m in sorted(all_results.items(), key=lambda x: x[1].get("RF_AUROC_eccrine_vs_hair", 0), reverse=True):
        if "error" in m:
            result_text += f"| {name} | FAILED | | | | |\n"
            continue
        result_text += (
            f"| {name} | "
            f"{m.get('ARI', 0):.3f} | "
            f"{m.get('AMI', 0):.3f} | "
            f"{m.get('ASW', 0):.3f} | "
            f"{m.get('RF_AUROC_eccrine_vs_hair', 0):.4f} | "
            f"{m.get('LR_AUROC_eccrine_vs_hair', 0):.4f} |\n"
        )

    with open(os.path.join(PROJECT_DIR, "results.md"), "a") as f:
        f.write(f"\n\n---\n\n### Proper Benchmark: Cryptic Fate Resolution\n")
        f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(result_text)

    print(f"\nResults appended to {os.path.join(PROJECT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
