"""
Evaluation metrics for PRISM.

Includes clustering metrics (ARI, AMI, NMI, ASW, Cohen's kappa),
classification metrics (F1, AUROC, AUPRC, ECE, Brier score),
neighborhood metrics (kNN purity, LISI, batch mixing entropy),
embedding quality metrics (trustworthiness), and
marker gene recovery (Precision@k, Recall@k).
"""

import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    brier_score_loss,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold


def compute_cohens_kappa(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> float:
    """Compute Cohen's kappa between KMeans clusters and true labels.

    Cohen's kappa measures inter-rater agreement adjusted for chance.
    Ranges from -1 (complete disagreement) to 1 (perfect agreement).

    Args:
        embeddings: (N, d) cell embeddings
        true_labels: (N,) ground truth labels
        n_clusters: number of clusters for KMeans (defaults to number of unique labels)

    Returns:
        Cohen's kappa score
    """
    from sklearn.cluster import KMeans

    n_k = n_clusters or len(np.unique(true_labels))
    kmeans = KMeans(n_clusters=n_k, random_state=42, n_init=10)
    pred = kmeans.fit_predict(embeddings)
    return cohen_kappa_score(true_labels, pred)


def compute_knn_purity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute kNN purity at various k values.

    For each cell, computes the fraction of its k nearest neighbors
    that share the same label. Higher values indicate better separation.

    Args:
        embeddings: (N, d) cell embeddings
        labels: (N,) cell labels
        k_values: list of k values to evaluate (default: [10, 50, 100])

    Returns:
        Dict with kNN_purity@k for each k
    """
    if k_values is None:
        k_values = [10, 50, 100]

    n_cells = len(labels)
    max_k = min(max(k_values), n_cells - 1)

    # Fit once with the largest k needed
    nn = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto")
    nn.fit(embeddings)
    # indices includes the cell itself at position 0
    distances, indices = nn.kneighbors(embeddings)

    results = {}
    for k in k_values:
        if k >= n_cells:
            results[f"kNN_purity@{k}"] = np.nan
            continue
        # Exclude self (index 0), take next k neighbors
        neighbor_indices = indices[:, 1:k + 1]
        neighbor_labels = labels[neighbor_indices]
        # Fraction of neighbors with same label as the cell
        same_label = (neighbor_labels == labels[:, np.newaxis]).mean(axis=1)
        results[f"kNN_purity@{k}"] = float(same_label.mean())

    return results


def compute_lisi(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 30,
) -> float:
    """Compute Local Inverse Simpson Index (LISI).

    For each cell, finds k nearest neighbors, computes the Simpson index
    from label frequencies in the neighborhood, and takes the inverse.
    The result is averaged across all cells.

    Higher LISI means more label mixing in neighborhoods.
    - For iLISI (batch labels): higher = better batch integration
    - For cLISI (cell-type labels): lower = better cell-type separation

    Args:
        embeddings: (N, d) cell embeddings
        labels: (N,) labels (batch labels for iLISI, cell-type for cLISI)
        k: number of nearest neighbors (default: 30)

    Returns:
        Mean LISI score across all cells
    """
    n_cells = len(labels)
    k_actual = min(k, n_cells - 1)

    nn = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="auto")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    # Exclude self (index 0)
    neighbor_indices = indices[:, 1:k_actual + 1]

    unique_labels = np.unique(labels)
    lisi_scores = np.zeros(n_cells)

    for i in range(n_cells):
        neighbor_labs = labels[neighbor_indices[i]]
        # Compute frequency of each label
        simpson = 0.0
        for lab in unique_labels:
            freq = np.sum(neighbor_labs == lab) / k_actual
            simpson += freq ** 2
        # Inverse Simpson index
        lisi_scores[i] = 1.0 / simpson if simpson > 0 else 1.0

    return float(lisi_scores.mean())


def compute_ilisi_clisi(
    embeddings: np.ndarray,
    fate_labels: np.ndarray,
    batch_labels: Optional[np.ndarray] = None,
    k: int = 30,
) -> Dict[str, float]:
    """Compute both iLISI and cLISI.

    Args:
        embeddings: (N, d) cell embeddings
        fate_labels: (N,) cell-type / fate labels
        batch_labels: (N,) batch labels (optional, needed for iLISI)
        k: number of nearest neighbors

    Returns:
        Dict with cLISI and optionally iLISI
    """
    results = {}

    # cLISI: cell-type LISI (lower = better separation)
    results["cLISI"] = compute_lisi(embeddings, fate_labels, k=k)

    # iLISI: integration LISI (higher = better batch mixing)
    if batch_labels is not None:
        results["iLISI"] = compute_lisi(embeddings, batch_labels, k=k)

    return results


def compute_batch_mixing_entropy(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    k: int = 50,
) -> float:
    """Compute batch mixing entropy.

    For each cell's kNN neighborhood, computes the Shannon entropy of
    the batch label distribution. Higher entropy = better batch mixing.
    Normalized by log(n_batches) so result is in [0, 1].

    Args:
        embeddings: (N, d) cell embeddings
        batch_labels: (N,) batch labels
        k: number of nearest neighbors (default: 50)

    Returns:
        Mean normalized Shannon entropy of batch labels in kNN neighborhoods
    """
    n_cells = len(batch_labels)
    k_actual = min(k, n_cells - 1)

    nn = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="auto")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    # Exclude self
    neighbor_indices = indices[:, 1:k_actual + 1]

    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches <= 1:
        return 0.0

    max_entropy = np.log(n_batches)
    entropies = np.zeros(n_cells)

    for i in range(n_cells):
        neighbor_labs = batch_labels[neighbor_indices[i]]
        entropy = 0.0
        for batch in unique_batches:
            freq = np.sum(neighbor_labs == batch) / k_actual
            if freq > 0:
                entropy -= freq * np.log(freq)
        # Normalize to [0, 1]
        entropies[i] = entropy / max_entropy

    return float(entropies.mean())


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence, then computes the weighted average
    of |accuracy - confidence| per bin.

    For multiclass, uses the predicted class probability as confidence.

    Args:
        y_true: (N,) true labels
        y_prob: (N, C) predicted probabilities from classifier
        n_bins: number of calibration bins (default: 10)

    Returns:
        ECE score (lower is better calibrated)
    """
    # Get predicted class and its confidence
    pred_classes = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (pred_classes == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = confidences[mask].mean()
        avg_accuracy = accuracies[mask].mean()
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute Brier score averaged over classes (OVR for multiclass).

    The Brier score measures the mean squared difference between predicted
    probabilities and the actual outcomes. Lower is better.

    Args:
        y_true: (N,) true labels
        y_prob: (N, C) predicted probabilities from classifier

    Returns:
        Mean Brier score across classes
    """
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)

    if n_classes == 2:
        return float(brier_score_loss(y_true, y_prob[:, 1]))

    # Multiclass: OVR Brier score averaged across classes
    brier_scores = []
    for i, cls in enumerate(unique_classes):
        y_binary = (y_true == cls).astype(int)
        brier_scores.append(brier_score_loss(y_binary, y_prob[:, i]))

    return float(np.mean(brier_scores))


def compute_trustworthiness(
    X_original: np.ndarray,
    embeddings: np.ndarray,
    n_neighbors: int = 15,
) -> float:
    """Compute trustworthiness of the embedding.

    Trustworthiness measures whether nearest neighbors in the embedding
    were also near in the original high-dimensional space. Score in [0, 1],
    higher is better.

    Args:
        X_original: (N, p) original high-dimensional expression matrix
        embeddings: (N, d) low-dimensional embeddings
        n_neighbors: number of neighbors to consider (default: 15)

    Returns:
        Trustworthiness score
    """
    # Limit n_neighbors to valid range
    max_k = min(n_neighbors, len(embeddings) - 2)
    if max_k < 1:
        return np.nan
    return float(sklearn_trustworthiness(X_original, embeddings, n_neighbors=max_k))


def compute_clustering_metrics(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    known_threshold: int = 2,
) -> Dict[str, float]:
    """Compute clustering quality metrics.

    Args:
        embeddings: (N, d) cell embeddings
        true_labels: (N,) ground truth labels
        predicted_labels: (N,) predicted cluster labels (if None, compute via Leiden)
        n_clusters: number of clusters (for kmeans if no predicted labels)
        known_threshold: labels >= this are considered known fates

    Returns:
        Dict with ARI, AMI, NMI, ASW, Cohen's kappa scores
    """
    # Filter to cells with known labels
    known_mask = true_labels >= known_threshold
    if known_mask.sum() < 10:
        known_mask = true_labels >= 0

    emb = embeddings[known_mask]
    true = true_labels[known_mask]

    if predicted_labels is not None:
        pred = predicted_labels[known_mask]
    else:
        # Use KMeans clustering
        from sklearn.cluster import KMeans
        n_k = n_clusters or len(np.unique(true))
        kmeans = KMeans(n_clusters=n_k, random_state=42, n_init=10)
        pred = kmeans.fit_predict(emb)

    metrics = {
        "ARI": adjusted_rand_score(true, pred),
        "AMI": adjusted_mutual_info_score(true, pred),
        "NMI": normalized_mutual_info_score(true, pred),
    }

    # Cohen's kappa
    metrics["Cohens_kappa"] = cohen_kappa_score(true, pred)

    # Silhouette score (on true labels, in embedding space)
    if len(np.unique(true)) >= 2 and len(emb) > len(np.unique(true)):
        try:
            metrics["ASW"] = silhouette_score(emb, true)
        except Exception:
            metrics["ASW"] = 0.0

    return metrics


def compute_classification_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    known_threshold: int = 2,
) -> Dict[str, float]:
    """Compute classification accuracy using cross-validated RF and LR.

    Evaluates how well the learned embeddings separate cell fates
    using standard classifiers.

    Args:
        embeddings: (N, d) cell embeddings
        labels: (N,) fate labels
        n_folds: number of CV folds
        seed: random seed
        known_threshold: labels >= this are considered known fates

    Returns:
        Dict with F1-macro, AUROC, AUPRC for each classifier
    """
    # Filter to cells with known fate labels
    known_mask = labels >= known_threshold
    if known_mask.sum() < 20:
        return {"error": "too_few_known_labels"}

    X = embeddings[known_mask]
    y = labels[known_mask]

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return {"error": "single_class"}

    # Binary or multi-class classification
    is_binary = len(unique_classes) == 2

    if is_binary:
        y_binary = (y == unique_classes[0]).astype(int)
    else:
        y_binary = y  # multi-class

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_probs = cross_val_predict(rf, X, y_binary, cv=cv, method="predict_proba")
    rf_preds = cross_val_predict(rf, X, y_binary, cv=cv, method="predict")

    results["RF_F1_macro"] = f1_score(y_binary, rf_preds, average="macro")
    if is_binary:
        results["RF_AUROC"] = roc_auc_score(y_binary, rf_probs[:, 1])
        results["RF_AUPRC"] = average_precision_score(y_binary, rf_probs[:, 1])
    else:
        results["RF_AUROC"] = roc_auc_score(y_binary, rf_probs, multi_class="ovr", average="macro")
        results["RF_AUPRC"] = 0.0  # AUPRC not well-defined for multi-class

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr_probs = cross_val_predict(lr, X, y_binary, cv=cv, method="predict_proba")
    lr_preds = cross_val_predict(lr, X, y_binary, cv=cv, method="predict")

    results["LR_F1_macro"] = f1_score(y_binary, lr_preds, average="macro")
    if is_binary:
        results["LR_AUROC"] = roc_auc_score(y_binary, lr_probs[:, 1])
        results["LR_AUPRC"] = average_precision_score(y_binary, lr_probs[:, 1])
    else:
        results["LR_AUROC"] = roc_auc_score(y_binary, lr_probs, multi_class="ovr", average="macro")
        results["LR_AUPRC"] = 0.0

    return results


def compute_marker_recovery(
    ranked_genes: List[str],
    known_eccrine_markers: List[str] = None,
    known_hair_markers: List[str] = None,
    k_values: List[int] = None,
    known_markers: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """Compute marker gene recovery metrics.

    Evaluates Precision@k and Recall@k for known markers
    among top-ranked PRISM-Resolve discriminators.

    Args:
        ranked_genes: Genes ranked by PRISM-Resolve (descending importance)
        known_eccrine_markers: Known eccrine marker genes (legacy, use known_markers instead)
        known_hair_markers: Known hair marker genes (legacy, use known_markers instead)
        k_values: Values of k for Precision@k
        known_markers: Dict mapping fate names to marker gene lists

    Returns:
        Dict with Precision@k and Recall@k for each k
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    # Build all_known from known_markers dict or legacy params
    if known_markers is not None:
        all_known = set()
        for markers in known_markers.values():
            all_known.update(markers)
    else:
        if known_eccrine_markers is None:
            known_eccrine_markers = ["En1", "Trpv6", "Dkk4", "Foxi1", "Defb6"]
        if known_hair_markers is None:
            known_hair_markers = ["Lhx2", "Sox9", "Wnt10b", "Shh", "Edar"]
        all_known = set(known_eccrine_markers + known_hair_markers)
    ranked_lower = [g.lower() for g in ranked_genes]
    known_lower = {g.lower() for g in all_known}

    results = {}

    for k in k_values:
        top_k = set(ranked_lower[:k])
        recovered = top_k & known_lower

        precision = len(recovered) / k if k > 0 else 0
        recall = len(recovered) / len(known_lower) if len(known_lower) > 0 else 0

        results[f"Precision@{k}"] = precision
        results[f"Recall@{k}"] = recall

        # Also track which markers were found
        results[f"recovered@{k}"] = [
            g for g in all_known if g.lower() in recovered
        ]

    return results


def compute_all_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ranked_genes: Optional[List[str]] = None,
    method_name: str = "PRISM",
    known_threshold: int = 2,
    known_eccrine_markers: Optional[List[str]] = None,
    known_hair_markers: Optional[List[str]] = None,
    known_markers: Optional[Dict[str, List[str]]] = None,
    batch_labels: Optional[np.ndarray] = None,
    X_original: Optional[np.ndarray] = None,
) -> Dict:
    """Compute all PRISM evaluation metrics.

    Args:
        embeddings: (N, d) cell embeddings
        labels: (N,) fate labels
        ranked_genes: Genes ranked by importance (optional)
        method_name: Name of the method being evaluated
        known_threshold: labels >= this are considered known fates
        known_eccrine_markers: Legacy eccrine markers
        known_hair_markers: Legacy hair markers
        known_markers: Dict mapping fate names to marker gene lists
        batch_labels: (N,) batch labels for integration metrics (optional)
        X_original: (N, p) original expression matrix for trustworthiness (optional)

    Returns:
        Dict with all computed metrics
    """
    results = {"method": method_name}

    # Clustering metrics (includes ARI, AMI, NMI, ASW, Cohen's kappa)
    clustering = compute_clustering_metrics(embeddings, labels, known_threshold=known_threshold)
    results.update(clustering)

    # Classification metrics
    classification = compute_classification_metrics(embeddings, labels, known_threshold=known_threshold)
    results.update(classification)

    # kNN purity
    known_mask = labels >= known_threshold
    if known_mask.sum() < 10:
        known_mask = labels >= 0
    emb_known = embeddings[known_mask]
    labels_known = labels[known_mask]

    try:
        knn_purity = compute_knn_purity(emb_known, labels_known)
        results.update(knn_purity)
    except Exception:
        pass

    # LISI metrics (cLISI always, iLISI if batch_labels provided)
    try:
        batch_known = batch_labels[known_mask] if batch_labels is not None else None
        lisi = compute_ilisi_clisi(emb_known, labels_known, batch_labels=batch_known)
        results.update(lisi)
    except Exception:
        pass

    # Batch mixing entropy (if batch_labels provided)
    if batch_labels is not None:
        try:
            batch_known = batch_labels[known_mask]
            results["batch_mixing_entropy"] = compute_batch_mixing_entropy(
                emb_known, batch_known
            )
        except Exception:
            pass

    # ECE and Brier score from RF classifier
    if "error" not in classification:
        try:
            y = labels[known_mask]
            unique_classes = np.unique(y)
            is_binary = len(unique_classes) == 2
            if is_binary:
                y_binary = (y == unique_classes[0]).astype(int)
            else:
                y_binary = y

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_probs = cross_val_predict(
                rf, emb_known, y_binary, cv=cv, method="predict_proba"
            )
            results["RF_ECE"] = compute_ece(y_binary, rf_probs)
            results["RF_Brier"] = compute_brier_score(y_binary, rf_probs)
        except Exception:
            pass

    # Trustworthiness (if original expression matrix provided)
    if X_original is not None:
        try:
            X_orig_known = X_original[known_mask]
            results["trustworthiness"] = compute_trustworthiness(
                X_orig_known, emb_known
            )
        except Exception:
            pass

    # Marker recovery (if genes provided)
    if ranked_genes:
        recovery = compute_marker_recovery(
            ranked_genes,
            known_eccrine_markers=known_eccrine_markers,
            known_hair_markers=known_hair_markers,
            known_markers=known_markers,
        )
        results.update(recovery)

    return results


def compute_extended_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    batch_labels: Optional[np.ndarray] = None,
    X_original: Optional[np.ndarray] = None,
    method_name: str = "",
) -> Dict[str, float]:
    """Compute ALL metrics (existing + new) in a single call.

    Convenience function that computes the full suite of PRISM evaluation
    metrics including clustering, classification, neighborhood, calibration,
    and embedding quality metrics.

    Args:
        embeddings: (N, d) cell embeddings
        labels: (N,) fate/cell-type labels (integer-encoded)
        batch_labels: (N,) batch labels for integration metrics (optional).
            If provided, computes iLISI and batch mixing entropy.
        X_original: (N, p) original high-dimensional expression matrix (optional).
            If provided, computes trustworthiness.
        method_name: Name of the method being evaluated (e.g., "PRISM", "PCA")

    Returns:
        Dict with all computed metrics:
            Clustering: ARI, AMI, NMI, ASW, Cohens_kappa
            Classification: RF_F1_macro, RF_AUROC, RF_AUPRC, LR_F1_macro, LR_AUROC, LR_AUPRC
            Neighborhood: kNN_purity@10, kNN_purity@50, kNN_purity@100
            LISI: cLISI (always), iLISI (if batch_labels)
            Batch: batch_mixing_entropy (if batch_labels)
            Calibration: RF_ECE, RF_Brier
            Embedding: trustworthiness (if X_original)
    """
    results = {"method": method_name}

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        results["error"] = "single_class"
        return results

    # --- Clustering metrics ---
    from sklearn.cluster import KMeans

    n_k = len(unique_labels)
    kmeans = KMeans(n_clusters=n_k, random_state=42, n_init=10)
    pred = kmeans.fit_predict(embeddings)

    results["ARI"] = adjusted_rand_score(labels, pred)
    results["AMI"] = adjusted_mutual_info_score(labels, pred)
    results["NMI"] = normalized_mutual_info_score(labels, pred)
    results["Cohens_kappa"] = cohen_kappa_score(labels, pred)

    if len(unique_labels) >= 2 and len(embeddings) > len(unique_labels):
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
            results["batch_mixing_entropy"] = compute_batch_mixing_entropy(
                embeddings, batch_labels
            )
        except Exception:
            pass

    # --- Classification metrics (RF + LR) with ECE and Brier ---
    is_binary = len(unique_labels) == 2
    if is_binary:
        y_eval = (labels == unique_labels[0]).astype(int)
    else:
        y_eval = labels

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_probs = cross_val_predict(rf, embeddings, y_eval, cv=cv, method="predict_proba")
        rf_preds = cross_val_predict(rf, embeddings, y_eval, cv=cv, method="predict")

        results["RF_F1_macro"] = f1_score(y_eval, rf_preds, average="macro")
        if is_binary:
            results["RF_AUROC"] = roc_auc_score(y_eval, rf_probs[:, 1])
            results["RF_AUPRC"] = average_precision_score(y_eval, rf_probs[:, 1])
        else:
            results["RF_AUROC"] = roc_auc_score(y_eval, rf_probs, multi_class="ovr", average="macro")
            results["RF_AUPRC"] = 0.0

        # ECE and Brier from RF
        results["RF_ECE"] = compute_ece(y_eval, rf_probs)
        results["RF_Brier"] = compute_brier_score(y_eval, rf_probs)

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr_probs = cross_val_predict(lr, embeddings, y_eval, cv=cv, method="predict_proba")
        lr_preds = cross_val_predict(lr, embeddings, y_eval, cv=cv, method="predict")

        results["LR_F1_macro"] = f1_score(y_eval, lr_preds, average="macro")
        if is_binary:
            results["LR_AUROC"] = roc_auc_score(y_eval, lr_probs[:, 1])
            results["LR_AUPRC"] = average_precision_score(y_eval, lr_probs[:, 1])
        else:
            results["LR_AUROC"] = roc_auc_score(y_eval, lr_probs, multi_class="ovr", average="macro")
            results["LR_AUPRC"] = 0.0

    except Exception:
        pass

    # --- Trustworthiness ---
    if X_original is not None:
        try:
            results["trustworthiness"] = compute_trustworthiness(
                X_original, embeddings
            )
        except Exception:
            pass

    return results
