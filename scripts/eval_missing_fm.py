#!/usr/bin/env python
"""Evaluate scGPT and Geneformer for the 6 systems that were missing them.

Adds rows to the existing full_evaluation_results.csv.
"""

import os, sys, json, warnings
import numpy as np
import anndata as ad
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
    cohen_kappa_score, silhouette_score, f1_score, roc_auc_score, average_precision_score
)
from prism.utils.metrics import (
    compute_knn_purity, compute_ilisi_clisi, compute_batch_mixing_entropy,
    compute_ece, compute_brier_score
)


def get_batch_labels(adata, config_path):
    """Get batch labels for iLISI/batch entropy computation."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    condition_key = cfg["system"].get("condition_key", "genotype")

    for key in [condition_key, "sample", "batch", "library", "plate", "condition", "stage", "timepoint", "cell_type"]:
        if key in adata.obs.columns and adata.obs[key].nunique() > 1:
            return adata.obs[key].astype(str).values
    return None


def compute_metrics_fast(embeddings, labels, batch_labels=None, X_original=None, method_name=""):
    """Compute all metrics with n_jobs=-1 and single cross_val_predict."""
    results = {"method": method_name}

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        results["error"] = "single_class"
        return results

    # Handle NaN/Inf
    if np.any(~np.isfinite(embeddings)):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Clustering ---
    n_k = len(unique_labels)
    kmeans = KMeans(n_clusters=n_k, random_state=42, n_init=10)
    pred = kmeans.fit_predict(embeddings)

    results["ARI"] = adjusted_rand_score(labels, pred)
    results["AMI"] = adjusted_mutual_info_score(labels, pred)
    results["NMI"] = normalized_mutual_info_score(labels, pred)
    results["Cohens_kappa"] = cohen_kappa_score(labels, pred)

    try:
        results["ASW"] = silhouette_score(embeddings, labels)
    except Exception:
        results["ASW"] = 0.0

    # --- kNN purity ---
    try:
        knn_purity = compute_knn_purity(embeddings, labels)
        results.update(knn_purity)
    except Exception:
        pass

    # --- LISI ---
    try:
        lisi = compute_ilisi_clisi(embeddings, labels, batch_labels=batch_labels)
        results.update(lisi)
    except Exception:
        pass

    # --- Batch mixing entropy ---
    if batch_labels is not None:
        try:
            results["batch_mixing_entropy"] = compute_batch_mixing_entropy(embeddings, batch_labels)
        except Exception:
            pass

    # --- Classification (single cross_val_predict, derive preds from probs) ---
    is_binary = len(unique_labels) == 2
    y_eval = (labels == unique_labels[0]).astype(int) if is_binary else labels

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Random Forest - use n_jobs=-1 and only one cross_val_predict call
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_probs = cross_val_predict(rf, embeddings, y_eval, cv=cv, method="predict_proba")
        rf_preds = np.argmax(rf_probs, axis=1)
        # Map back to original labels for preds
        label_classes = np.unique(y_eval)
        rf_preds_mapped = label_classes[rf_preds]

        results["RF_F1_macro"] = f1_score(y_eval, rf_preds_mapped, average="macro")
        if is_binary:
            results["RF_AUROC"] = roc_auc_score(y_eval, rf_probs[:, 1])
            results["RF_AUPRC"] = average_precision_score(y_eval, rf_probs[:, 1])
        else:
            results["RF_AUROC"] = roc_auc_score(y_eval, rf_probs, multi_class="ovr", average="macro")
            results["RF_AUPRC"] = 0.0

        results["RF_ECE"] = compute_ece(y_eval, rf_probs)
        results["RF_Brier"] = compute_brier_score(y_eval, rf_probs)

        # Logistic Regression - n_jobs=-1 not available, but much faster than RF
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr_probs = cross_val_predict(lr, embeddings, y_eval, cv=cv, method="predict_proba")
        lr_preds = label_classes[np.argmax(lr_probs, axis=1)]

        results["LR_F1_macro"] = f1_score(y_eval, lr_preds, average="macro")
        if is_binary:
            results["LR_AUROC"] = roc_auc_score(y_eval, lr_probs[:, 1])
            results["LR_AUPRC"] = average_precision_score(y_eval, lr_probs[:, 1])
        else:
            results["LR_AUROC"] = roc_auc_score(y_eval, lr_probs, multi_class="ovr", average="macro")
            results["LR_AUPRC"] = 0.0
    except Exception as e:
        print(f"    Classification failed: {e}")

    # --- Trustworthiness ---
    if X_original is not None:
        try:
            from sklearn.manifold import trustworthiness as tw
            results["trustworthiness"] = tw(X_original, embeddings, n_neighbors=15)
        except Exception:
            pass

    return results


MISSING_SYSTEMS = {
    "oligo": {
        "adata": "data/processed/oligo/adata_processed.h5ad",
        "config": "configs/oligo.yaml",
        "scgpt": "data/processed/oligo/scgpt_embeddings.npy",
        "geneformer": "data/processed/oligo/geneformer_embeddings.npy",
    },
    "paul": {
        "adata": "data/processed/paul/adata_processed.h5ad",
        "config": "configs/paul.yaml",
        "scgpt": "data/processed/paul/scgpt_embeddings.npy",
        "geneformer": "data/processed/paul/geneformer_embeddings.npy",
    },
    "nestorowa": {
        "adata": "data/processed/nestorowa/adata_processed.h5ad",
        "config": "configs/nestorowa.yaml",
        "scgpt": "data/processed/nestorowa/scgpt_embeddings.npy",
        "geneformer": "data/processed/nestorowa/geneformer_embeddings.npy",
    },
    "sadefeldman": {
        "adata": "data/processed/sadefeldman/adata_processed.h5ad",
        "config": "configs/sadefeldman.yaml",
        "scgpt": "data/processed/sadefeldman/scgpt_embeddings.npy",
        "geneformer": "data/processed/sadefeldman/geneformer_embeddings.npy",
    },
    "tirosh_melanoma": {
        "adata": "data/processed/tirosh_melanoma/adata_processed.h5ad",
        "config": "configs/tirosh_melanoma.yaml",
        "scgpt": "data/processed/tirosh_melanoma/scgpt_embeddings.npy",
        "geneformer": "data/processed/tirosh_melanoma/geneformer_embeddings.npy",
    },
    "neftel_gbm": {
        "adata": "data/processed/neftel_gbm/adata_processed.h5ad",
        "config": "configs/neftel_gbm.yaml",
        "scgpt": "data/processed/neftel_gbm/scgpt_embeddings.npy",
        "geneformer": "data/processed/neftel_gbm/geneformer_embeddings.npy",
    },
}


def main():
    print("=" * 60)
    print("  Evaluating scGPT + Geneformer for 6 missing systems")
    print("=" * 60)

    # Load existing CSV
    csv_path = "data/processed/full_evaluation_results.csv"
    df = pd.read_csv(csv_path)
    print(f"Existing CSV: {len(df)} rows")

    new_rows = []

    for sys_name, info in MISSING_SYSTEMS.items():
        print(f"\n{'='*50}")
        print(f"  {sys_name}")
        print(f"{'='*50}")

        adata = ad.read_h5ad(info["adata"])
        print(f"  {adata.n_obs} cells")

        # Get labels
        if "fate_int" in adata.obs.columns:
            labels = adata.obs["fate_int"].values.astype(int)
        else:
            labels = adata.obs["fate_label"].astype("category").cat.codes.values

        batch_labels = get_batch_labels(adata, info["config"])
        X_original = None
        if hasattr(adata.X, "toarray"):
            if adata.n_obs * adata.n_vars < 500_000_000:
                X_original = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            X_original = adata.X

        for model_name, emb_path in [("scGPT", info["scgpt"]), ("Geneformer", info["geneformer"])]:
            if not os.path.exists(emb_path):
                print(f"  {model_name}: no embeddings, skipping")
                continue

            emb = np.load(emb_path)
            if emb.shape[0] != adata.n_obs:
                print(f"  {model_name}: shape mismatch ({emb.shape[0]} vs {adata.n_obs})")
                continue

            print(f"  {model_name}: {emb.shape} ...")
            metrics = compute_metrics_fast(emb, labels, batch_labels, X_original, model_name)

            if "error" not in metrics:
                auroc = metrics.get("RF_AUROC", "N/A")
                ari = metrics.get("ARI", "N/A")
                print(f"    ARI={ari:.3f}  RF_AUROC={auroc:.3f}")
            else:
                print(f"    Error: {metrics['error']}")

            # Build row matching CSV columns
            row = {"system": sys_name, "method": model_name}
            for col in df.columns:
                if col in ("system", "method"):
                    continue
                row[col] = metrics.get(col, "")
            new_rows.append(row)

    # Append new rows to CSV
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"\nUpdated CSV: {len(df)} rows ({len(new_rows)} new)")

        # Also update combined JSON
        json_path = "data/processed/full_evaluation_results.json"
        combined = json.load(open(json_path))
        for _, row in pd.DataFrame(new_rows).iterrows():
            sys_name = row["system"]
            method = row["method"]
            if sys_name not in combined:
                combined[sys_name] = {}
            combined[sys_name][method] = {k: v for k, v in row.items()
                                          if k not in ("system",) and v != "" and pd.notna(v)}
        with open(json_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"Updated JSON: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("  Final RF_AUROC Summary")
    print(f"{'='*60}")
    df_final = pd.read_csv(csv_path)
    pivot = df_final.pivot_table(index="system", columns="method", values="RF_AUROC", aggfunc="first")
    method_order = ["PRISM", "PCA", "Harmony", "DiffMap", "UMAP", "scGPT", "Geneformer"]
    method_order = [m for m in method_order if m in pivot.columns]
    pivot = pivot[method_order]
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "---"))

    print("\nDone!")


if __name__ == "__main__":
    main()
