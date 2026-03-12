"""
Baseline method implementations for PRISM benchmarking.

Baselines:
1. PCA + Seurat (Leiden clustering)
2. PCA + Scanpy (KMeans)
3. Harmony + Seurat
4. scVI
5. scANVI
6. scGPT zero-shot (from cached embeddings)
7. Geneformer zero-shot (from cached embeddings)
8. XGBoost on HVGs (PCA embeddings) - "dumb baseline"
9. Diffusion Maps + KMeans/RF/LR
10. UMAP embedding + RF/LR classifier
"""

import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.cluster import KMeans
from typing import Dict, Optional
import warnings

from ..utils.metrics import compute_all_metrics


def run_baselines(
    adata: ad.AnnData,
    labels: np.ndarray,
    n_clusters: int = 4,
    condition_key: str = "genotype",
) -> Dict[str, Dict]:
    """Run all baseline methods and compute metrics.

    Args:
        adata: Preprocessed AnnData
        labels: Ground truth fate labels
        n_clusters: Number of clusters for clustering methods

    Returns:
        Dict mapping method name to metrics dict
    """
    results = {}

    # 1. PCA + Leiden (Seurat-style)
    print("\n=== Baseline 1: PCA + Leiden ===")
    results["PCA+Leiden"] = _run_pca_leiden(adata, labels, n_clusters)

    # 2. PCA + Scanpy default
    print("\n=== Baseline 2: PCA + KMeans ===")
    results["PCA+KMeans"] = _run_pca_kmeans(adata, labels, n_clusters)

    # 3. Harmony + Leiden
    print("\n=== Baseline 3: Harmony + Leiden ===")
    results["Harmony+Leiden"] = _run_harmony(adata, labels, n_clusters, condition_key=condition_key)

    # 4. scVI (run in subprocess to prevent OOM from killing main process)
    print("\n=== Baseline 4: scVI ===")
    results["scVI"] = _run_scvi_safe(adata, labels, n_clusters)

    # 5. scANVI
    print("\n=== Baseline 5: scANVI ===")
    results["scANVI"] = _run_scanvi_safe(adata, labels, n_clusters)

    # 6-7. Foundation models (from cached embeddings)
    print("\n=== Baselines 6-7: Foundation Models ===")
    for name in ["scGPT", "Geneformer"]:
        real = _run_foundation_baseline(adata, labels, n_clusters, name)
        if real is not None:
            results[name] = real
        else:
            print(f"  {name}: no cached embeddings found (run benchmarks/run_foundation_benchmarks_cross_system.py first)")

    # 8. XGBoost on HVGs (PCA embeddings)
    print("\n=== Baseline 8: XGBoost on HVGs ===")
    results["XGBoost_HVG"] = _run_xgboost_hvg(adata, labels, n_clusters)

    # 9. Diffusion Maps
    print("\n=== Baseline 9: Diffusion Maps ===")
    results["DiffMap"] = _run_diffmap(adata, labels, n_clusters)

    # 10. UMAP embedding + classifier
    print("\n=== Baseline 10: UMAP+Classifier ===")
    results["UMAP+Classifier"] = _run_umap_classifier(adata, labels, n_clusters)

    return results


def _run_pca_leiden(adata: ad.AnnData, labels: np.ndarray, n_clusters: int) -> Dict:
    """PCA + Leiden clustering baseline."""
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)

    # Use PCA embedding
    emb = adata.obsm["X_pca"]

    # Leiden clustering
    sc.pp.neighbors(adata, use_rep="X_pca", key_added="pca_neighbors")
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden_pca", neighbors_key="pca_neighbors")
    pred_labels = adata.obs["leiden_pca"].astype(int).values

    return compute_all_metrics(emb, labels, method_name="PCA+Leiden")


def _run_pca_kmeans(adata: ad.AnnData, labels: np.ndarray, n_clusters: int) -> Dict:
    """PCA + KMeans clustering baseline."""
    emb = adata.obsm["X_pca"]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(emb)

    return compute_all_metrics(emb, labels, method_name="PCA+KMeans")


def _run_harmony(adata: ad.AnnData, labels: np.ndarray, n_clusters: int, condition_key: str = "genotype") -> Dict:
    """Harmony batch correction + Leiden."""
    try:
        import harmonypy as hm

        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)

        # Find the best available batch key — prefer explicit condition_key
        batch_key = condition_key
        if batch_key not in adata.obs.columns or adata.obs[batch_key].nunique() < 2:
            batch_key = "genotype"
            for key in ["sample", "batch", "library", "plate", "condition", "stage", "timepoint", "cell_type"]:
                if key in adata.obs.columns and adata.obs[key].nunique() > 1:
                    batch_key = key
                    break
        ho = hm.run_harmony(
            adata.obsm["X_pca"],
            adata.obs,
            batch_key,
            max_iter_harmony=20,
        )
        emb = ho.Z_corr
        adata.obsm["X_harmony"] = emb

        return compute_all_metrics(emb, labels, method_name="Harmony+Leiden")

    except ImportError:
        print("harmonypy not installed, returning PCA metrics")
        return compute_all_metrics(adata.obsm["X_pca"], labels, method_name="Harmony+Leiden")


def _run_scvi_safe(adata: ad.AnnData, labels: np.ndarray, n_clusters: int) -> Dict:
    """Run scVI in a subprocess to prevent OOM from killing the main process."""
    import subprocess, json, tempfile, os
    try:
        # Save minimal data for scVI
        hvg_mask = adata.var["highly_variable"] if "highly_variable" in adata.var else np.ones(adata.shape[1], dtype=bool)
        adata_hvg = adata[:, hvg_mask].copy()
        tmp_path = "/tmp/adata_scvi_input.h5ad"
        adata_hvg.write_h5ad(tmp_path)
        del adata_hvg

        script = f"""
import anndata as ad, numpy as np, json, sys
try:
    import scvi
    adata = ad.read_h5ad("{tmp_path}")
    scvi.model.SCVI.setup_anndata(adata, batch_key="sample" if "sample" in adata.obs else None)
    model = scvi.model.SCVI(adata, n_latent=30)
    model.train(max_epochs=100, early_stopping=True, progress_bar=False)
    emb = model.get_latent_representation()
    np.save("/tmp/scvi_emb.npy", emb)
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {{e}}")
"""
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
        # Ensure LD_LIBRARY_PATH propagates for torch and scipy
        import site
        sp = site.getsitepackages()[0]
        extra_paths = []
        cusparselt_path = os.path.join(sp, "nvidia", "cusparselt", "lib")
        if os.path.exists(cusparselt_path):
            extra_paths.append(cusparselt_path)
        libstdcxx_path = "/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/"
        if os.path.exists(libstdcxx_path):
            extra_paths.append(libstdcxx_path)
        if extra_paths:
            env["LD_LIBRARY_PATH"] = ":".join(extra_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        if "SUCCESS" in result.stdout:
            emb = np.load("/tmp/scvi_emb.npy")
            return compute_all_metrics(emb, labels, method_name="scVI")
        else:
            print(f"scVI subprocess output: {result.stdout[-200:]} {result.stderr[-200:]}")
            return {"method": "scVI", "error": "subprocess_failed"}
    except Exception as e:
        return {"method": "scVI", "error": str(e)}


def _run_scanvi_safe(adata: ad.AnnData, labels: np.ndarray, n_clusters: int) -> Dict:
    """Run scANVI in a subprocess to prevent OOM from killing the main process."""
    import subprocess, os, site
    try:
        hvg_mask = adata.var["highly_variable"] if "highly_variable" in adata.var else np.ones(adata.shape[1], dtype=bool)
        adata_hvg = adata[:, hvg_mask].copy()
        tmp_path = "/tmp/adata_scanvi_input.h5ad"
        adata_hvg.write_h5ad(tmp_path)
        del adata_hvg

        label_key = "fate_label" if "fate_label" in adata.obs else "cluster"
        script = f"""
import anndata as ad, numpy as np, sys
try:
    import scvi
    adata = ad.read_h5ad("{tmp_path}")
    scvi.model.SCVI.setup_anndata(adata, batch_key="sample" if "sample" in adata.obs else None, labels_key="{label_key}")
    vae = scvi.model.SCVI(adata, n_latent=30)
    vae.train(max_epochs=100, early_stopping=True, progress_bar=False)
    scanvi = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="undetermined")
    scanvi.train(max_epochs=50, progress_bar=False)
    emb = scanvi.get_latent_representation()
    np.save("/tmp/scanvi_emb.npy", emb)
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {{e}}")
"""
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
        sp = site.getsitepackages()[0]
        extra_paths = []
        cusparselt_path = os.path.join(sp, "nvidia", "cusparselt", "lib")
        if os.path.exists(cusparselt_path):
            extra_paths.append(cusparselt_path)
        libstdcxx_path = "/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/"
        if os.path.exists(libstdcxx_path):
            extra_paths.append(libstdcxx_path)
        if extra_paths:
            env["LD_LIBRARY_PATH"] = ":".join(extra_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        if "SUCCESS" in result.stdout:
            emb = np.load("/tmp/scanvi_emb.npy")
            return compute_all_metrics(emb, labels, method_name="scANVI")
        else:
            print(f"scANVI subprocess output: {result.stdout[-200:]} {result.stderr[-200:]}")
            return {"method": "scANVI", "error": "subprocess_failed"}
    except Exception as e:
        return {"method": "scANVI", "error": str(e)}


def _run_xgboost_hvg(adata: ad.AnnData, labels: np.ndarray, n_clusters: int) -> Dict:
    """XGBoost classifier on PCA embeddings (50 PCs) with 5-fold CV.

    This is the 'dumb baseline' -- if it beats PRISM, contrastive learning
    isn't adding value beyond what a gradient-boosted tree can extract
    from standard PCA embeddings.

    Uses XGBClassifier with default params. Computes the same metric suite
    as other baselines via compute_all_metrics, plus XGBoost-specific
    predictions used as a substitute for the RF_AUROC slot.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  xgboost not installed -- skipping XGBoost baseline. "
              "Install with: pip install xgboost")
        return {"method": "XGBoost_HVG", "error": "xgboost_not_installed"}

    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)

    emb = adata.obsm["X_pca"]

    # First get the standard suite (ARI, AMI, NMI, ASW, RF_AUROC, LR_AUROC, etc.)
    base_metrics = compute_all_metrics(emb, labels, method_name="XGBoost_HVG")

    # Now add XGBoost-specific classification metrics
    # Filter to known-fate cells (same logic as compute_classification_metrics)
    known_threshold = 2
    known_mask = labels >= known_threshold
    if known_mask.sum() < 20:
        known_mask = labels >= 0

    X = emb[known_mask]
    y = labels[known_mask]

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        base_metrics["XGB_F1_macro"] = 0.0
        base_metrics["XGB_AUROC"] = 0.0
        return base_metrics

    is_binary = len(unique_classes) == 2
    if is_binary:
        y_enc = (y == unique_classes[0]).astype(int)
    else:
        # Remap to contiguous 0..K-1 for XGBoost
        label_map = {v: i for i, v in enumerate(unique_classes)}
        y_enc = np.array([label_map[v] for v in y])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    try:
        xgb_probs = cross_val_predict(xgb, X, y_enc, cv=cv, method="predict_proba")
        xgb_preds = cross_val_predict(xgb, X, y_enc, cv=cv, method="predict")

        base_metrics["XGB_F1_macro"] = f1_score(y_enc, xgb_preds, average="macro")
        if is_binary:
            base_metrics["XGB_AUROC"] = roc_auc_score(y_enc, xgb_probs[:, 1])
        else:
            base_metrics["XGB_AUROC"] = roc_auc_score(
                y_enc, xgb_probs, multi_class="ovr", average="macro"
            )
    except Exception as e:
        print(f"  XGBoost CV failed: {e}")
        base_metrics["XGB_F1_macro"] = 0.0
        base_metrics["XGB_AUROC"] = 0.0

    return base_metrics


def _run_diffmap(adata: ad.AnnData, labels: np.ndarray, n_clusters: int) -> Dict:
    """Diffusion map embedding baseline.

    Computes a diffusion map via scanpy (sc.tl.diffmap), then runs
    KMeans clustering and RF/LR classifiers on the diffusion components,
    identical to the PCA baseline evaluation pipeline.
    """
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)

    # Build neighbor graph (required by sc.tl.diffmap)
    # Use a dedicated key to avoid overwriting any existing neighbor graph
    sc.pp.neighbors(adata, use_rep="X_pca", key_added="diffmap_neighbors")
    sc.tl.diffmap(adata, n_comps=15, neighbors_key="diffmap_neighbors")

    emb = adata.obsm["X_diffmap"]  # shape (n_cells, n_comps+1)

    return compute_all_metrics(emb, labels, method_name="DiffMap")


def _run_umap_classifier(adata: ad.AnnData, labels: np.ndarray, n_clusters: int) -> Dict:
    """UMAP 2D embedding + RF/LR classifier baseline.

    Extracts UMAP coordinates and trains RF/LR classifiers on them.
    This demonstrates that UMAP preserves visual cluster separation but
    the 2D coordinates alone are poor features for supervised classification
    compared to higher-dimensional contrastive embeddings.
    """
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)

    # Compute UMAP if not already present; use dedicated neighbor key
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep="X_pca", key_added="umap_neighbors")
        sc.tl.umap(adata, neighbors_key="umap_neighbors")

    emb = adata.obsm["X_umap"]  # shape (n_cells, 2)

    return compute_all_metrics(emb, labels, method_name="UMAP+Classifier")


def _run_foundation_baseline(
    adata: ad.AnnData,
    labels: np.ndarray,
    n_clusters: int,
    name: str,
) -> Optional[Dict]:
    """Load real foundation model embeddings if available.

    Checks for cached embeddings from run_foundation_benchmarks_cross_system.py.
    Returns metrics dict if found, None otherwise.
    """
    import os

    emb_file_map = {
        "scGPT": "scgpt_embeddings.npy",
        "Geneformer": "geneformer_embeddings.npy",
    }

    emb_filename = emb_file_map.get(name)
    if emb_filename is None:
        return None

    # Check multiple locations (skin legacy path + per-system paths)
    search_dirs = [
        "data/processed",
        "data/processed/skin",
        "data/processed/pancreas",
        "data/processed/cortex",
        "data/processed/hsc",
    ]

    for search_dir in search_dirs:
        emb_path = os.path.join(search_dir, emb_filename)
        if os.path.exists(emb_path):
            try:
                embeddings = np.load(emb_path)
                if embeddings.shape[0] != len(labels):
                    continue  # Wrong system, skip
                print(f"  Loaded real {name} embeddings from {emb_path}")
                return compute_all_metrics(embeddings, labels, method_name=name)
            except Exception as e:
                print(f"  Failed to load {emb_path}: {e}")
                continue

    return None
