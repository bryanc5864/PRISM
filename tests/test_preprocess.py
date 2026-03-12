"""Tests for preprocessing pipeline."""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from prism.data.preprocess import preprocess_adata, assign_labels, assign_genotypes, split_data


class TestPreprocessAdata:
    def test_basic_preprocessing(self):
        """preprocess_adata runs on synthetic data without error."""
        np.random.seed(42)
        n_cells, n_genes = 200, 1000
        X = sp.random(n_cells, n_genes, density=0.3, format="csr", dtype=np.float32)
        X.data = np.abs(X.data) * 100

        adata = ad.AnnData(X=X)
        adata.var_names = pd.Index([f"Gene_{i}" for i in range(n_genes)])
        adata.obs_names = pd.Index([f"Cell_{i}" for i in range(n_cells)])
        adata.obs["sample"] = "S1"

        result = preprocess_adata(adata, min_genes=5, max_genes=5000, max_mito_pct=100, n_hvgs=200)

        assert result.shape[0] > 0
        assert "X_pca" in result.obsm
        assert "highly_variable" in result.var
        assert "normalized" in result.layers
        assert result.obsm["X_pca"].shape[1] == 50

    def test_hvg_selection(self):
        """HVG selection produces correct number of genes."""
        np.random.seed(42)
        n_cells, n_genes = 200, 1000
        X = sp.random(n_cells, n_genes, density=0.3, format="csr", dtype=np.float32)
        X.data = np.abs(X.data) * 100

        adata = ad.AnnData(X=X)
        adata.var_names = pd.Index([f"Gene_{i}" for i in range(n_genes)])
        adata.obs_names = pd.Index([f"Cell_{i}" for i in range(n_cells)])
        adata.obs["sample"] = "S1"

        result = preprocess_adata(adata, min_genes=5, max_genes=5000, max_mito_pct=100, n_hvgs=100)

        # Should have at least 100 HVGs (could have more with forced genes)
        assert result.var["highly_variable"].sum() >= 100


class TestAssignLabels:
    def test_hierarchical_strategy(self, small_adata):
        """Hierarchical label assignment produces expected columns."""
        result = assign_labels(
            small_adata,
            condition_key="genotype",
            conditions={"WT": 0, "KO": 1},
            label_strategy="hierarchical",
        )
        assert "fate_label" in result.obs.columns
        assert "fate_int" in result.obs.columns
        assert "cell_type" in result.obs.columns

    def test_flat_strategy(self, small_adata):
        """Flat label assignment produces expected columns."""
        result = assign_labels(
            small_adata,
            condition_key="genotype",
            conditions={"WT": 0, "KO": 1},
            fate_categories=["background", "undetermined", "fate_a", "fate_b"],
            marker_scores={"fate_a": ["Gene_10", "Gene_11"], "fate_b": ["Gene_20", "Gene_21"]},
            label_strategy="flat",
        )
        assert "fate_label" in result.obs.columns
        assert "fate_int" in result.obs.columns

    def test_annotation_strategy(self, small_adata):
        """Annotation-based label assignment maps correctly."""
        # Add annotation column
        small_adata.obs["cell_type_annotation"] = np.random.choice(
            ["Ery", "Neu", "B", "Undifferentiated"], size=small_adata.shape[0]
        )

        result = assign_labels(
            small_adata,
            condition_key="genotype",
            conditions={"WT": 0, "KO": 1},
            fate_categories=["background", "undetermined", "erythroid", "myeloid", "lymphoid"],
            label_strategy="annotation",
            annotation_key="cell_type_annotation",
            annotation_fate_map={"Ery": "erythroid", "Neu": "myeloid", "B": "lymphoid", "Undifferentiated": "undetermined"},
        )
        assert "fate_label" in result.obs.columns
        # All cells should have valid fate labels
        valid_fates = {"background", "undetermined", "erythroid", "myeloid", "lymphoid"}
        assert set(result.obs["fate_label"].unique()).issubset(valid_fates)

    def test_annotation_fallback_to_flat(self, small_adata):
        """Annotation strategy falls back to flat when annotation column is missing."""
        result = assign_labels(
            small_adata,
            condition_key="genotype",
            conditions={"WT": 0, "KO": 1},
            fate_categories=["background", "undetermined", "fate_a"],
            marker_scores={"fate_a": ["Gene_10"]},
            label_strategy="annotation",
            annotation_key="nonexistent_column",
            annotation_fate_map={},
        )
        assert "fate_label" in result.obs.columns

    def test_fate_int_encoding(self, small_adata):
        """fate_int correctly encodes fate_label."""
        cats = ["non_appendage", "undetermined", "eccrine", "hair"]
        result = assign_labels(
            small_adata,
            condition_key="genotype",
            conditions={"WT": 0, "KO": 1},
            fate_categories=cats,
            label_strategy="hierarchical",
        )
        # Each fate_label should map to the correct integer
        for i, cat in enumerate(cats):
            mask = result.obs["fate_label"] == cat
            if mask.any():
                assert (result.obs.loc[mask, "fate_int"] == i).all()


class TestAssignGenotypes:
    def test_basic_genotype_assignment(self, small_adata):
        """assign_genotypes maps samples to conditions."""
        result = assign_genotypes(
            small_adata,
            sample_condition_map={"S1": "WT", "S2": "WT", "S3": "KO", "S4": "KO"},
            condition_key="genotype",
        )
        assert "genotype" in result.obs.columns
        assert set(result.obs["genotype"].unique()) == {"WT", "KO"}


class TestSplitData:
    def test_split_sizes(self, small_adata):
        """Split produces correct proportions."""
        train, val, test = split_data(small_adata, train_frac=0.8, val_frac=0.1, test_frac=0.1)

        total = train.shape[0] + val.shape[0] + test.shape[0]
        assert total == small_adata.shape[0]
        # Allow some tolerance for rounding
        assert train.shape[0] > val.shape[0]
        assert train.shape[0] > test.shape[0]

    def test_no_overlap(self, small_adata):
        """Train/val/test sets don't overlap."""
        train, val, test = split_data(small_adata)

        train_idx = set(train.obs_names)
        val_idx = set(val.obs_names)
        test_idx = set(test.obs_names)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
