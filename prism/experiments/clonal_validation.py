"""
Clonal validation for PRISM using Weinreb et al. lineage tracing data.

Validates PRISM's fate assignments against ground-truth clonal barcodes:
1. Clonal fate concordance: do early PRISM predictions match late observed fates?
2. Clonal purity: fraction of clone-mates assigned same PRISM fate
3. Fate predictability: RF on early-timepoint PRISM embeddings predicts late fates
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Optional, Tuple
import anndata as ad


def compute_clonal_fate_concordance(
    adata: ad.AnnData,
    fate_col: str = "fate_label",
    time_col: str = "time_point",
    early_time: Optional[str] = None,
    late_time: Optional[str] = None,
) -> Dict[str, float]:
    """Compute concordance between early PRISM fate predictions and late observed fates.

    For clones present at both early and late timepoints, checks whether
    the PRISM fate assigned at the early timepoint matches the observed
    fate at the late timepoint (determined by majority vote of late cells).

    Args:
        adata: AnnData with clone_matrix in obsm, fate_col and time_col in obs
        fate_col: obs column with fate assignments
        time_col: obs column with time point labels
        early_time: label for early timepoint (auto-detected if None)
        late_time: label for late timepoint (auto-detected if None)

    Returns:
        Dict with concordance rate, n_clones tested, per-fate concordance
    """
    if "clone_matrix" not in adata.obsm:
        return {"error": "no_clone_matrix"}
    if time_col not in adata.obs.columns:
        return {"error": f"no_{time_col}_column"}

    clone_mat = adata.obsm["clone_matrix"]
    if sp.issparse(clone_mat):
        clone_mat = clone_mat.toarray()

    times = adata.obs[time_col].values
    fates = adata.obs[fate_col].values

    # Auto-detect early/late timepoints
    unique_times = sorted(pd.Series(times).dropna().unique())
    if len(unique_times) < 2:
        return {"error": "need_at_least_2_timepoints", "unique_times": list(unique_times)}

    if early_time is None:
        early_time = unique_times[0]
    if late_time is None:
        late_time = unique_times[-1]

    early_mask = times == early_time
    late_mask = times == late_time

    # Find clones with cells in both timepoints
    early_clone_presence = clone_mat[early_mask].sum(axis=0) > 0
    late_clone_presence = clone_mat[late_mask].sum(axis=0) > 0
    shared_clones = np.where(early_clone_presence & late_clone_presence)[0]

    if len(shared_clones) == 0:
        return {"error": "no_shared_clones", "n_early": int(early_mask.sum()),
                "n_late": int(late_mask.sum())}

    concordant = 0
    total = 0
    per_fate_concordant = {}
    per_fate_total = {}

    for clone_idx in shared_clones:
        # Get cells from this clone at each timepoint
        early_cells = np.where(early_mask & (clone_mat[:, clone_idx] > 0))[0]
        late_cells = np.where(late_mask & (clone_mat[:, clone_idx] > 0))[0]

        if len(early_cells) == 0 or len(late_cells) == 0:
            continue

        # Early prediction: majority vote of PRISM fate assignments
        early_fates = fates[early_cells]
        early_pred = pd.Series(early_fates).mode().iloc[0]

        # Late observed: majority vote of observed fates
        late_fates = fates[late_cells]
        late_obs = pd.Series(late_fates).mode().iloc[0]

        total += 1
        if early_pred == late_obs:
            concordant += 1

        # Per-fate tracking
        for fate in [early_pred, late_obs]:
            if fate not in per_fate_total:
                per_fate_total[fate] = 0
                per_fate_concordant[fate] = 0

        per_fate_total[late_obs] += 1
        if early_pred == late_obs:
            per_fate_concordant[late_obs] += 1

    concordance_rate = concordant / total if total > 0 else 0.0

    per_fate_rates = {}
    for fate in per_fate_total:
        if per_fate_total[fate] > 0:
            per_fate_rates[fate] = per_fate_concordant[fate] / per_fate_total[fate]

    return {
        "concordance_rate": concordance_rate,
        "n_concordant": concordant,
        "n_tested_clones": total,
        "n_shared_clones": len(shared_clones),
        "early_time": str(early_time),
        "late_time": str(late_time),
        "per_fate_concordance": per_fate_rates,
    }


def compute_clonal_purity(
    adata: ad.AnnData,
    fate_col: str = "fate_label",
    min_clone_size: int = 2,
) -> Dict[str, float]:
    """Compute fraction of clone-mates assigned the same PRISM fate.

    For each clone with >= min_clone_size cells, computes the fraction
    of cell pairs that share the same PRISM fate assignment. Compares
    against a random baseline.

    Args:
        adata: AnnData with clone_matrix in obsm
        fate_col: obs column with fate assignments
        min_clone_size: minimum cells per clone to include

    Returns:
        Dict with mean purity, random baseline, and per-clone stats
    """
    if "clone_matrix" not in adata.obsm:
        return {"error": "no_clone_matrix"}

    clone_mat = adata.obsm["clone_matrix"]
    if sp.issparse(clone_mat):
        clone_mat = clone_mat.toarray()

    fates = adata.obs[fate_col].values
    n_clones = clone_mat.shape[1]

    purities = []
    clone_sizes = []

    for clone_idx in range(n_clones):
        cells = np.where(clone_mat[:, clone_idx] > 0)[0]
        if len(cells) < min_clone_size:
            continue

        clone_fates = fates[cells]
        # Purity = fraction of pairs with same fate
        n = len(clone_fates)
        same_pairs = 0
        total_pairs = n * (n - 1) / 2
        for i in range(n):
            for j in range(i + 1, n):
                if clone_fates[i] == clone_fates[j]:
                    same_pairs += 1

        purity = same_pairs / total_pairs if total_pairs > 0 else 1.0
        purities.append(purity)
        clone_sizes.append(n)

    if not purities:
        return {"error": "no_clones_with_enough_cells", "min_clone_size": min_clone_size}

    # Random baseline: expected purity if fates assigned randomly
    fate_counts = pd.Series(fates).value_counts(normalize=True)
    random_purity = float((fate_counts ** 2).sum())

    return {
        "mean_purity": float(np.mean(purities)),
        "median_purity": float(np.median(purities)),
        "std_purity": float(np.std(purities)),
        "random_baseline": random_purity,
        "purity_over_random": float(np.mean(purities)) - random_purity,
        "n_clones_tested": len(purities),
        "mean_clone_size": float(np.mean(clone_sizes)),
    }


def compute_fate_predictability(
    adata: ad.AnnData,
    embedding_key: str = "X_prism",
    fate_col: str = "fate_label",
    time_col: str = "time_point",
    early_time: Optional[str] = None,
    late_time: Optional[str] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Train RF on early-timepoint PRISM embeddings, predict late-timepoint fates.

    Uses clonal linkage as ground truth: for cells at the early timepoint,
    their "true" late fate is determined by the majority fate of their
    clone-mates at the late timepoint. Then trains RF on early embeddings
    to predict these fates.

    Args:
        adata: AnnData with clone_matrix, embeddings, timepoints
        embedding_key: obsm key for embeddings
        fate_col: obs column with fate assignments
        time_col: obs column with timepoint labels
        early_time: label for early timepoint
        late_time: label for late timepoint

    Returns:
        Dict with RF accuracy, AUROC, and per-fate metrics
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if "clone_matrix" not in adata.obsm or embedding_key not in adata.obsm:
        return {"error": "missing_clone_matrix_or_embeddings"}
    if time_col not in adata.obs.columns:
        return {"error": f"no_{time_col}_column"}

    clone_mat = adata.obsm["clone_matrix"]
    if sp.issparse(clone_mat):
        clone_mat = clone_mat.toarray()

    times = adata.obs[time_col].values
    fates = adata.obs[fate_col].values

    unique_times = sorted(pd.Series(times).dropna().unique())
    if early_time is None:
        early_time = unique_times[0]
    if late_time is None:
        late_time = unique_times[-1]

    early_mask = times == early_time
    late_mask = times == late_time

    # For each early cell, determine its "true" late fate via clonal linkage
    early_indices = np.where(early_mask)[0]
    early_embeddings = []
    early_true_fates = []

    for cell_idx in early_indices:
        # Find which clone(s) this cell belongs to
        cell_clones = np.where(clone_mat[cell_idx] > 0)[0]
        if len(cell_clones) == 0:
            continue

        # Find late-timepoint clone-mates
        late_clone_cells = []
        for clone_idx in cell_clones:
            late_cells = np.where(late_mask & (clone_mat[:, clone_idx] > 0))[0]
            late_clone_cells.extend(late_cells)

        if not late_clone_cells:
            continue

        # Majority fate of late clone-mates = ground truth
        late_fates = fates[late_clone_cells]
        true_fate = pd.Series(late_fates).mode().iloc[0]

        early_embeddings.append(adata.obsm[embedding_key][cell_idx])
        early_true_fates.append(true_fate)

    if len(early_embeddings) < 20:
        return {"error": "too_few_linked_cells", "n_linked": len(early_embeddings)}

    X = np.array(early_embeddings)
    y = np.array(early_true_fates)

    # Encode labels as integers
    unique_fates = sorted(set(y))
    fate_to_int = {f: i for i, f in enumerate(unique_fates)}
    y_int = np.array([fate_to_int[f] for f in y])

    n_classes = len(unique_fates)
    n_folds_actual = min(n_folds, min(np.bincount(y_int)))
    if n_folds_actual < 2:
        n_folds_actual = 2

    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    cv = StratifiedKFold(n_splits=n_folds_actual, shuffle=True, random_state=seed)

    preds = cross_val_predict(rf, X, y_int, cv=cv, method="predict")
    probs = cross_val_predict(rf, X, y_int, cv=cv, method="predict_proba")

    accuracy = accuracy_score(y_int, preds)
    f1 = f1_score(y_int, preds, average="macro")

    if n_classes == 2:
        auroc = roc_auc_score(y_int, probs[:, 1])
    elif n_classes > 2:
        auroc = roc_auc_score(y_int, probs, multi_class="ovr", average="macro")
    else:
        auroc = 0.0

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "auroc": float(auroc),
        "n_early_cells_linked": len(early_embeddings),
        "n_fates": n_classes,
        "fate_names": unique_fates,
        "n_folds": n_folds_actual,
    }


def run_clonal_validation(
    adata: ad.AnnData,
    fate_col: str = "fate_label",
    time_col: str = "time_point",
    embedding_key: str = "X_prism",
    save_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """Run all clonal validation analyses.

    Args:
        adata: AnnData with clone_matrix, embeddings, fate labels, timepoints
        fate_col: obs column with fate assignments
        time_col: obs column with timepoint labels
        embedding_key: obsm key for embeddings
        save_dir: optional directory to save results

    Returns:
        Dict with results from all three analyses
    """
    import os

    print("\n=== Clonal Validation ===")

    results = {}

    # 1. Clonal fate concordance
    print("\n--- Clonal Fate Concordance ---")
    concordance = compute_clonal_fate_concordance(adata, fate_col=fate_col, time_col=time_col)
    results["concordance"] = concordance
    if "error" not in concordance:
        print(f"  Concordance rate: {concordance['concordance_rate']:.3f}")
        print(f"  Tested clones: {concordance['n_tested_clones']}")
        print(f"  Per-fate: {concordance['per_fate_concordance']}")
    else:
        print(f"  Skipped: {concordance['error']}")

    # 2. Clonal purity
    print("\n--- Clonal Purity ---")
    purity = compute_clonal_purity(adata, fate_col=fate_col)
    results["purity"] = purity
    if "error" not in purity:
        print(f"  Mean purity: {purity['mean_purity']:.3f}")
        print(f"  Random baseline: {purity['random_baseline']:.3f}")
        print(f"  Purity over random: {purity['purity_over_random']:.3f}")
        print(f"  Clones tested: {purity['n_clones_tested']}")
    else:
        print(f"  Skipped: {purity['error']}")

    # 3. Fate predictability
    print("\n--- Fate Predictability ---")
    predictability = compute_fate_predictability(
        adata, embedding_key=embedding_key, fate_col=fate_col, time_col=time_col,
    )
    results["predictability"] = predictability
    if "error" not in predictability:
        print(f"  Accuracy: {predictability['accuracy']:.3f}")
        print(f"  F1 macro: {predictability['f1_macro']:.3f}")
        print(f"  AUROC: {predictability['auroc']:.3f}")
        print(f"  Linked early cells: {predictability['n_early_cells_linked']}")
    else:
        print(f"  Skipped: {predictability['error']}")

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import json
        # Convert non-serializable types
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                kk: (vv if isinstance(vv, (int, float, str, bool, list, dict)) else str(vv))
                for kk, vv in v.items()
            }
        with open(os.path.join(save_dir, "clonal_validation.json"), "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Saved results to {save_dir}/clonal_validation.json")

    return results
