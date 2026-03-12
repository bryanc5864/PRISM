"""Tests for evaluation metrics."""

import numpy as np
import pytest

from prism.utils.metrics import (
    compute_clustering_metrics,
    compute_classification_metrics,
    compute_marker_recovery,
    compute_all_metrics,
)


class TestClusteringMetrics:
    def test_perfect_clustering(self):
        """Perfect embeddings should give high ARI."""
        np.random.seed(42)
        n = 200
        # Two well-separated clusters
        emb = np.vstack([
            np.random.randn(100, 10) + 5,
            np.random.randn(100, 10) - 5,
        ])
        labels = np.array([2] * 100 + [3] * 100)

        metrics = compute_clustering_metrics(emb, labels, known_threshold=2)

        assert "ARI" in metrics
        assert "AMI" in metrics
        assert "NMI" in metrics
        assert "ASW" in metrics
        assert metrics["ARI"] > 0.8
        assert metrics["ASW"] > 0.5

    def test_random_clustering(self):
        """Random embeddings should give low ARI."""
        np.random.seed(42)
        emb = np.random.randn(200, 10)
        labels = np.array([2] * 100 + [3] * 100)

        metrics = compute_clustering_metrics(emb, labels, known_threshold=2)
        # ARI near 0 for random
        assert metrics["ARI"] < 0.3

    def test_known_threshold_filtering(self):
        """Only cells with labels >= threshold are used."""
        np.random.seed(42)
        emb = np.vstack([
            np.random.randn(50, 10) + 5,   # label 0 (below threshold)
            np.random.randn(50, 10) - 5,    # label 2
            np.random.randn(50, 10),         # label 3
        ])
        labels = np.array([0] * 50 + [2] * 50 + [3] * 50)

        metrics = compute_clustering_metrics(emb, labels, known_threshold=2)
        assert "ARI" in metrics


class TestClassificationMetrics:
    def test_separable_embeddings(self):
        """Well-separated embeddings should give high AUROC."""
        np.random.seed(42)
        emb = np.vstack([
            np.random.randn(100, 10) + 5,
            np.random.randn(100, 10) - 5,
        ])
        labels = np.array([2] * 100 + [3] * 100)

        metrics = compute_classification_metrics(emb, labels, known_threshold=2)

        assert "RF_AUROC" in metrics
        assert "RF_F1_macro" in metrics
        assert "LR_AUROC" in metrics
        assert metrics["RF_AUROC"] > 0.9

    def test_multiclass(self):
        """Multi-class classification uses OVR AUROC."""
        np.random.seed(42)
        emb = np.vstack([
            np.random.randn(80, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.random.randn(80, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.random.randn(80, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0]),
        ])
        labels = np.array([2] * 80 + [3] * 80 + [4] * 80)

        metrics = compute_classification_metrics(emb, labels, known_threshold=2)

        assert "RF_AUROC" in metrics
        assert metrics["RF_AUROC"] > 0.8

    def test_too_few_labels(self):
        """Returns error with too few known labels."""
        emb = np.random.randn(10, 10)
        labels = np.array([0] * 10)  # all below threshold

        metrics = compute_classification_metrics(emb, labels, known_threshold=2)
        assert "error" in metrics


class TestMarkerRecovery:
    def test_perfect_recovery(self):
        """All markers in top-k gives precision=1."""
        ranked = ["En1", "Trpv6", "Dkk4", "Lhx2", "Sox9", "Random1", "Random2"]
        known_a = ["En1", "Trpv6", "Dkk4"]
        known_b = ["Lhx2", "Sox9"]

        recovery = compute_marker_recovery(ranked, known_a, known_b, k_values=[5, 10])

        assert recovery["Precision@5"] == 1.0
        assert recovery["Recall@5"] == 1.0

    def test_partial_recovery(self):
        """Partial overlap gives correct precision and recall."""
        ranked = ["Random1", "En1", "Random2", "Lhx2", "Random3"]
        known_a = ["En1", "Trpv6"]
        known_b = ["Lhx2", "Sox9"]

        recovery = compute_marker_recovery(ranked, known_a, known_b, k_values=[5])

        # 2 out of 5 in top-5
        assert recovery["Precision@5"] == 2 / 5
        # 2 out of 4 known markers recovered
        assert recovery["Recall@5"] == 2 / 4

    def test_known_markers_dict(self):
        """known_markers dict works correctly."""
        ranked = ["En1", "Lhx2", "Random1"]
        recovery = compute_marker_recovery(
            ranked,
            known_markers={"a": ["En1"], "b": ["Lhx2"]},
            k_values=[2],
        )
        assert recovery["Precision@2"] == 1.0
        assert recovery["Recall@2"] == 1.0


class TestComputeAllMetrics:
    def test_combines_metrics(self):
        """compute_all_metrics combines clustering + classification."""
        np.random.seed(42)
        emb = np.vstack([
            np.random.randn(100, 10) + 5,
            np.random.randn(100, 10) - 5,
        ])
        labels = np.array([2] * 100 + [3] * 100)

        metrics = compute_all_metrics(emb, labels, method_name="test")

        assert metrics["method"] == "test"
        assert "ARI" in metrics
        assert "RF_AUROC" in metrics
