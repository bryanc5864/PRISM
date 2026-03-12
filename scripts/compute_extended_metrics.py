#!/usr/bin/env python -u
"""
Compute extended metrics on all PRISM-trained systems.

For each system, loads the processed adata, extracts PRISM embeddings,
and computes: kNN purity, LISI (cLISI + iLISI), Cohen's kappa, ECE,
Brier score, and trustworthiness.

Also computes metrics for baseline embeddings (PCA, scGPT, Geneformer)
where available.
"""

import json
import sys
import os
import warnings
import traceback
import time

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.utils.metrics import (
    compute_knn_purity,
    compute_ilisi_clisi,
    compute_batch_mixing_entropy,
    compute_trustworthiness,
    compute_ece,
    compute_brier_score,
)
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Flush all prints
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---- System definitions ----
SYSTEMS = [
    ("skin", "data/processed/adata_processed.h5ad", "genotype"),
    ("pancreas", "data/processed/pancreas/adata_processed.h5ad", "genotype"),
    ("cortex", "data/processed/cortex/adata_processed.h5ad", "stage"),
    ("hsc", "data/processed/hsc/adata_processed.h5ad", "genotype"),
    ("intestine", "data/processed/intestine/adata_processed.h5ad", "batch"),
    ("cardiac", "data/processed/cardiac/adata_processed.h5ad", "library"),
    ("neural_crest", "data/processed/neural_crest/adata_processed.h5ad", "stage"),
    ("thcell", "data/processed/thcell/adata_processed.h5ad", "condition"),
    ("lung", "data/processed/lung/adata_processed.h5ad", "timepoint"),
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Subsample thresholds
MAX_CELLS_CLASSIFICATION = 20000  # RF cross-val is O(n^2) with n_estimators
MAX_CELLS_KNN = 30000             # kNN is O(n*k)
MAX_CELLS_TRUSTWORTHINESS = 10000 # trustworthiness pairwise distances
MAX_CELLS_LISI = 30000            # LISI is O(n*k)


def get_output_dir(system_name):
    if system_name == "skin":
        return os.path.join(BASE_DIR, "data", "processed")
    return os.path.join(BASE_DIR, "data", "processed", system_name)


def get_X_dense_subset(adata, indices=None):
    """Extract dense expression matrix for a subset of cells."""
    X = adata.X
    if indices is not None:
        X = X[indices]
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def find_batch_key(adata, preferred_key):
    if preferred_key in adata.obs.columns:
        return preferred_key
    for key in ["genotype", "sample", "condition", "batch", "timepoint",
                "stage", "region", "library", "donor", "experiment"]:
        if key in adata.obs.columns:
            return key
    return None


def encode_labels(series):
    vals = series.values
    if hasattr(vals, "codes"):
        return vals.codes.astype(int)
    uniq = np.unique(vals.astype(str))
    mapping = {v: i for i, v in enumerate(uniq)}
    return np.array([mapping[str(v)] for v in vals], dtype=int)


def stratified_subsample(labels, max_n, seed=42):
    """Return indices for a stratified subsample."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n <= max_n:
        return np.arange(n)

    unique, counts = np.unique(labels, return_counts=True)
    # Proportional sampling
    fracs = counts / counts.sum()
    per_class = np.maximum((fracs * max_n).astype(int), 1)
    # Adjust to hit target
    while per_class.sum() > max_n:
        per_class[np.argmax(per_class)] -= 1
    while per_class.sum() < max_n:
        per_class[np.argmin(per_class / counts)] += 1

    indices = []
    for cls, n_cls in zip(unique, per_class):
        cls_idx = np.where(labels == cls)[0]
        chosen = rng.choice(cls_idx, min(n_cls, len(cls_idx)), replace=False)
        indices.append(chosen)
    return np.sort(np.concatenate(indices))


def compute_metrics_for_embedding(embeddings, labels, batch_labels, X_orig_fn,
                                   method_name, n_cells):
    """Compute all extended metrics for one embedding method.

    X_orig_fn: callable that returns dense X for given indices, or None.
    """
    t0 = time.time()
    results = {"method": method_name}

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    if n_classes < 2:
        results["error"] = "single_class"
        return results

    # --- 1. Clustering (KMeans) - fast, run on full data ---
    print(f"      Clustering...", end="")
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    pred = kmeans.fit_predict(embeddings)
    results["ARI"] = float(adjusted_rand_score(labels, pred))
    results["AMI"] = float(adjusted_mutual_info_score(labels, pred))
    results["NMI"] = float(normalized_mutual_info_score(labels, pred))
    results["Cohens_kappa"] = float(cohen_kappa_score(labels, pred))
    try:
        results["ASW"] = float(silhouette_score(embeddings, labels))
    except Exception:
        results["ASW"] = 0.0
    print(f" done ({time.time()-t0:.1f}s)")

    # --- 2. kNN purity (subsample if large) ---
    t1 = time.time()
    print(f"      kNN purity...", end="")
    if n_cells > MAX_CELLS_KNN:
        idx = stratified_subsample(labels, MAX_CELLS_KNN)
        knn_res = compute_knn_purity(embeddings[idx], labels[idx])
    else:
        knn_res = compute_knn_purity(embeddings, labels)
    results.update(knn_res)
    print(f" done ({time.time()-t1:.1f}s)")

    # --- 3. LISI (subsample if large) ---
    t1 = time.time()
    print(f"      LISI...", end="")
    if n_cells > MAX_CELLS_LISI:
        idx = stratified_subsample(labels, MAX_CELLS_LISI)
        bl = batch_labels[idx] if batch_labels is not None else None
        lisi_res = compute_ilisi_clisi(embeddings[idx], labels[idx], batch_labels=bl)
    else:
        lisi_res = compute_ilisi_clisi(embeddings, labels, batch_labels=batch_labels)
    results.update(lisi_res)
    print(f" done ({time.time()-t1:.1f}s)")

    # --- 4. Batch mixing entropy ---
    if batch_labels is not None:
        t1 = time.time()
        print(f"      Batch entropy...", end="")
        if n_cells > MAX_CELLS_KNN:
            idx = stratified_subsample(labels, MAX_CELLS_KNN)
            results["batch_mixing_entropy"] = float(
                compute_batch_mixing_entropy(embeddings[idx], batch_labels[idx])
            )
        else:
            results["batch_mixing_entropy"] = float(
                compute_batch_mixing_entropy(embeddings, batch_labels)
            )
        print(f" done ({time.time()-t1:.1f}s)")

    # --- 5. Classification (RF) with ECE and Brier ---
    t1 = time.time()
    print(f"      RF classification...", end="")
    is_binary = n_classes == 2
    if is_binary:
        y_eval = (labels == unique_labels[0]).astype(int)
    else:
        y_eval = labels

    # Subsample for classification
    if n_cells > MAX_CELLS_CLASSIFICATION:
        idx = stratified_subsample(labels, MAX_CELLS_CLASSIFICATION)
        emb_cls = embeddings[idx]
        y_cls = y_eval[idx] if is_binary else labels[idx]
        if is_binary:
            y_cls = (labels[idx] == unique_labels[0]).astype(int)
    else:
        emb_cls = embeddings
        y_cls = y_eval

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_probs = cross_val_predict(rf, emb_cls, y_cls, cv=cv, method="predict_proba")
        rf_preds = np.argmax(rf_probs, axis=1)
        # Map back to actual class labels
        classes = np.unique(y_cls)
        rf_preds_mapped = classes[rf_preds]

        results["RF_F1_macro"] = float(f1_score(y_cls, rf_preds_mapped, average="macro"))
        if is_binary:
            results["RF_AUROC"] = float(roc_auc_score(y_cls, rf_probs[:, 1]))
        else:
            results["RF_AUROC"] = float(roc_auc_score(y_cls, rf_probs, multi_class="ovr", average="macro"))

        results["RF_ECE"] = float(compute_ece(y_cls, rf_probs))
        results["RF_Brier"] = float(compute_brier_score(y_cls, rf_probs))
    except Exception as e:
        print(f" FAILED: {e}", end="")
    print(f" done ({time.time()-t1:.1f}s)")

    # --- 6. Trustworthiness (subsample to 10K) ---
    if X_orig_fn is not None:
        t1 = time.time()
        print(f"      Trustworthiness...", end="")
        try:
            if n_cells > MAX_CELLS_TRUSTWORTHINESS:
                idx = stratified_subsample(labels, MAX_CELLS_TRUSTWORTHINESS)
                X_sub = X_orig_fn(idx)
                emb_sub = embeddings[idx]
            else:
                X_sub = X_orig_fn(None)
                emb_sub = embeddings
            results["trustworthiness"] = float(compute_trustworthiness(X_sub, emb_sub))
            del X_sub
        except Exception as e:
            print(f" FAILED: {e}", end="")
            results["trustworthiness"] = None
        print(f" done ({time.time()-t1:.1f}s)")

    print(f"      Total: {time.time()-t0:.1f}s")
    return results


def process_system(system_name, data_path, condition_key):
    full_path = os.path.join(BASE_DIR, data_path)
    if not os.path.exists(full_path):
        print(f"  [SKIP] Data file not found: {full_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  Processing: {system_name}")
    print(f"{'='*60}")

    import anndata as ad

    print(f"  Loading {full_path}...")
    adata = ad.read_h5ad(full_path)
    n_cells = adata.shape[0]
    print(f"  Loaded: {n_cells} cells x {adata.shape[1]} genes")
    print(f"  obs columns: {list(adata.obs.columns[:20])}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")

    # --- Labels ---
    if "fate_int" in adata.obs.columns:
        labels = adata.obs["fate_int"].values.astype(int)
    elif "fate" in adata.obs.columns:
        labels = encode_labels(adata.obs["fate"])
    else:
        print(f"  [SKIP] No fate labels")
        return None

    unique_labels = np.unique(labels)
    print(f"  Labels: {len(unique_labels)} unique: {unique_labels}")

    # --- Batch ---
    batch_key = find_batch_key(adata, condition_key)
    batch_labels = None
    if batch_key is not None:
        n_uniq = len(adata.obs[batch_key].unique())
        print(f"  Batch key: '{batch_key}' ({n_uniq} values)")
        if n_uniq > 1:
            batch_labels = encode_labels(adata.obs[batch_key])
        else:
            print(f"    Only 1 batch value, skipping iLISI/batch metrics")

    # --- X_orig accessor (lazy, returns dense for given indices) ---
    def get_X_orig(indices):
        return get_X_dense_subset(adata, indices)

    out_dir = get_output_dir(system_name)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    # ---- PRISM ----
    if "X_prism" in adata.obsm:
        prism_emb = np.asarray(adata.obsm["X_prism"], dtype=np.float32)
        print(f"\n  [PRISM] shape={prism_emb.shape}")
        try:
            m = compute_metrics_for_embedding(
                prism_emb, labels, batch_labels, get_X_orig, "PRISM", n_cells)
            all_results["PRISM"] = m
            print(f"    => ARI={m.get('ARI','?'):.4f} RF_AUROC={m.get('RF_AUROC','?'):.4f} "
                  f"kNN@10={m.get('kNN_purity@10','?'):.4f} ECE={m.get('RF_ECE','?'):.4f} "
                  f"Brier={m.get('RF_Brier','?'):.4f} Trust={m.get('trustworthiness','?')}")
        except Exception as e:
            print(f"    [ERROR] {e}")
            traceback.print_exc()

    # ---- PCA ----
    if "X_pca" in adata.obsm:
        pca_emb = np.asarray(adata.obsm["X_pca"][:, :min(50, adata.obsm["X_pca"].shape[1])],
                             dtype=np.float32)
        print(f"\n  [PCA] shape={pca_emb.shape}")
        try:
            m = compute_metrics_for_embedding(
                pca_emb, labels, batch_labels, get_X_orig, "PCA", n_cells)
            all_results["PCA"] = m
            print(f"    => ARI={m.get('ARI','?'):.4f} RF_AUROC={m.get('RF_AUROC','?'):.4f} "
                  f"kNN@10={m.get('kNN_purity@10','?'):.4f} ECE={m.get('RF_ECE','?'):.4f} "
                  f"Brier={m.get('RF_Brier','?'):.4f} Trust={m.get('trustworthiness','?')}")
        except Exception as e:
            print(f"    [ERROR] {e}")
            traceback.print_exc()

    # ---- scGPT ----
    scgpt_path = os.path.join(out_dir, "scgpt_embeddings.npy")
    if system_name == "skin":
        scgpt_path = os.path.join(BASE_DIR, "data", "processed", "scgpt_embeddings.npy")
    if os.path.exists(scgpt_path):
        scgpt_emb = np.load(scgpt_path).astype(np.float32)
        if scgpt_emb.shape[0] == n_cells:
            print(f"\n  [scGPT] shape={scgpt_emb.shape}")
            try:
                m = compute_metrics_for_embedding(
                    scgpt_emb, labels, batch_labels, get_X_orig, "scGPT", n_cells)
                all_results["scGPT"] = m
                print(f"    => ARI={m.get('ARI','?'):.4f} RF_AUROC={m.get('RF_AUROC','?'):.4f} "
                      f"kNN@10={m.get('kNN_purity@10','?'):.4f} ECE={m.get('RF_ECE','?'):.4f} "
                      f"Brier={m.get('RF_Brier','?'):.4f} Trust={m.get('trustworthiness','?')}")
            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
        else:
            print(f"  [scGPT] SKIP: shape mismatch {scgpt_emb.shape[0]} vs {n_cells}")
    else:
        print(f"  [scGPT] not found")

    # ---- Geneformer ----
    gf_path = os.path.join(out_dir, "geneformer_embeddings.npy")
    if system_name == "skin":
        gf_path = os.path.join(BASE_DIR, "data", "processed", "geneformer_embeddings.npy")
    if os.path.exists(gf_path):
        gf_emb = np.load(gf_path).astype(np.float32)
        if gf_emb.shape[0] == n_cells:
            print(f"\n  [Geneformer] shape={gf_emb.shape}")
            try:
                m = compute_metrics_for_embedding(
                    gf_emb, labels, batch_labels, get_X_orig, "Geneformer", n_cells)
                all_results["Geneformer"] = m
                print(f"    => ARI={m.get('ARI','?'):.4f} RF_AUROC={m.get('RF_AUROC','?'):.4f} "
                      f"kNN@10={m.get('kNN_purity@10','?'):.4f} ECE={m.get('RF_ECE','?'):.4f} "
                      f"Brier={m.get('RF_Brier','?'):.4f} Trust={m.get('trustworthiness','?')}")
            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
        else:
            print(f"  [Geneformer] SKIP: shape mismatch {gf_emb.shape[0]} vs {n_cells}")
    else:
        print(f"  [Geneformer] not found")

    # --- Save ---
    output_file = os.path.join(out_dir, "extended_metrics.json")

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return obj

    with open(output_file, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n  Saved: {output_file}")

    del adata
    import gc; gc.collect()
    return all_results


def print_summary_table(all_system_results):
    print("\n\n" + "=" * 120)
    print("EXTENDED METRICS SUMMARY TABLE")
    print("=" * 120)

    key_metrics = [
        "ARI", "RF_AUROC", "kNN_purity@10", "kNN_purity@50",
        "cLISI", "iLISI", "Cohens_kappa", "RF_ECE", "RF_Brier",
        "trustworthiness", "ASW", "batch_mixing_entropy",
    ]
    methods = ["PRISM", "PCA", "scGPT", "Geneformer"]

    for metric in key_metrics:
        print(f"\n--- {metric} ---")
        header = f"{'System':<16}"
        for method in methods:
            header += f"{method:>14}"
        print(header)
        print("-" * (16 + 14 * len(methods)))

        for sname, results in all_system_results.items():
            if results is None:
                continue
            row = f"{sname:<16}"
            for method in methods:
                if method in results and metric in results[method]:
                    val = results[method][metric]
                    if val is not None:
                        row += f"{val:>14.4f}"
                    else:
                        row += f"{'N/A':>14}"
                else:
                    row += f"{'--':>14}"
            print(row)

    # Advantage table
    print(f"\n\n--- PRISM vs Baselines (RF_AUROC advantage) ---")
    print(f"{'System':<16}{'vs PCA':>12}{'vs scGPT':>12}{'vs GF':>12}")
    print("-" * 52)
    for sname, results in all_system_results.items():
        if results is None:
            continue
        prism_auroc = results.get("PRISM", {}).get("RF_AUROC")
        if prism_auroc is None:
            continue
        row = f"{sname:<16}"
        for method in ["PCA", "scGPT", "Geneformer"]:
            bl = results.get(method, {}).get("RF_AUROC")
            if bl is not None:
                row += f"{prism_auroc - bl:>+12.4f}"
            else:
                row += f"{'--':>12}"
        print(row)


def main():
    print("Computing extended metrics for all PRISM-trained systems")
    print(f"Base directory: {BASE_DIR}")
    t_start = time.time()

    all_system_results = {}
    for system_name, data_path, condition_key in SYSTEMS:
        try:
            results = process_system(system_name, data_path, condition_key)
            all_system_results[system_name] = results
        except Exception as e:
            print(f"\n  [FATAL] {system_name} failed: {e}")
            traceback.print_exc()
            all_system_results[system_name] = None

    print_summary_table(all_system_results)

    # Save combined summary
    summary_path = os.path.join(BASE_DIR, "data", "processed", "extended_metrics_summary.json")

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return obj

    valid = {k: convert(v) for k, v in all_system_results.items() if v is not None}
    with open(summary_path, "w") as f:
        json.dump(valid, f, indent=2)
    print(f"\nCombined summary saved: {summary_path}")
    print(f"Total time: {time.time()-t_start:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
