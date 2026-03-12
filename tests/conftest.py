"""Shared test fixtures for PRISM test suite."""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest


@pytest.fixture
def small_adata():
    """Create a small synthetic AnnData for testing (100 cells, 500 genes, 2 conditions, 3 fates)."""
    np.random.seed(42)
    n_cells = 100
    n_genes = 500

    X = sp.random(n_cells, n_genes, density=0.3, format="csr", dtype=np.float32)
    X.data = np.abs(X.data) * 10  # positive counts

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    # Include some known marker genes
    for i, name in enumerate(["Krt14", "Krt5", "En1", "Lhx2", "Sox9"]):
        if i < n_genes:
            gene_names[i] = name

    cell_names = [f"Cell_{i}" for i in range(n_cells)]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )

    # Add metadata
    conditions = np.array(["WT"] * 50 + ["KO"] * 50)
    samples = np.array(["S1"] * 25 + ["S2"] * 25 + ["S3"] * 25 + ["S4"] * 25)
    fate_labels = np.array(
        ["non_appendage"] * 30 + ["undetermined"] * 20 +
        ["eccrine"] * 25 + ["hair"] * 25
    )
    fate_ints = np.array([0] * 30 + [1] * 20 + [2] * 25 + [3] * 25)

    adata.obs["genotype"] = conditions
    adata.obs["sample"] = samples
    adata.obs["fate_label"] = fate_labels
    adata.obs["fate_int"] = fate_ints
    adata.obs["cluster"] = fate_labels

    # Add HVG flag
    adata.var["highly_variable"] = True

    # Add PCA embedding
    adata.obsm["X_pca"] = np.random.randn(n_cells, 50).astype(np.float32)

    return adata


@pytest.fixture
def small_adata_with_clones(small_adata):
    """Small AnnData with clone matrix and time points for clonal validation."""
    n_cells = small_adata.shape[0]
    n_clones = 10

    # Create clone matrix: each clone has 5-15 cells
    clone_mat = np.zeros((n_cells, n_clones), dtype=np.float32)
    np.random.seed(123)
    for c in range(n_clones):
        n_members = np.random.randint(5, 15)
        members = np.random.choice(n_cells, size=n_members, replace=False)
        clone_mat[members, c] = 1.0

    small_adata.obsm["clone_matrix"] = sp.csr_matrix(clone_mat)

    # Add time points: first half early, second half late
    times = np.array(["day2"] * 50 + ["day6"] * 50)
    small_adata.obs["time_point"] = times

    # Add PRISM embedding
    small_adata.obsm["X_prism"] = np.random.randn(n_cells, 128).astype(np.float32)

    return small_adata


@pytest.fixture
def skin_config():
    """Skin system config for testing."""
    from prism.config import SKIN_CONFIG
    return SKIN_CONFIG
