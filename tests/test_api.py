"""Tests for PRISM high-level API."""

import os
import tempfile
import numpy as np
import pytest

from prism.api import PRISM
from prism.config import SystemConfig, SKIN_CONFIG


class TestPRISMInit:
    def test_init_with_adata(self, small_adata):
        """PRISM initializes with AnnData."""
        model = PRISM(small_adata, condition_key="genotype")
        assert model.adata is small_adata
        assert model.condition_key == "genotype"
        assert model.system_config.name == "custom"

    def test_init_with_config(self, small_adata, skin_config):
        """PRISM initializes with SystemConfig."""
        model = PRISM(small_adata, condition_key="genotype", config=skin_config)
        assert model.system_config.name == "skin"
        assert model.system_config.condition_key == "genotype"

    def test_init_with_yaml(self, small_adata):
        """PRISM initializes from YAML system config."""
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "skin.yaml")
        if os.path.exists(yaml_path):
            model = PRISM(small_adata, condition_key="genotype", system=yaml_path)
            assert model.system_config.name == "skin"

    def test_state_flags(self, small_adata):
        """Initial state flags are False."""
        model = PRISM(small_adata, condition_key="genotype")
        assert not model._is_preprocessed
        assert not model._is_fitted
        assert not model._is_resolved
        assert not model._is_traced


class TestPRISMSaveLoad:
    def test_save_load_roundtrip(self, small_adata, skin_config):
        """Save/load preserves core data."""
        model = PRISM(small_adata, condition_key="genotype", config=skin_config)
        model._is_preprocessed = True

        # Add fake embeddings and DE results
        small_adata.obsm["X_prism"] = np.random.randn(small_adata.shape[0], 128).astype(np.float32)
        model._fate_probs = np.random.rand(small_adata.shape[0], 3).astype(np.float32)

        import pandas as pd
        model._de_results = pd.DataFrame({
            "gene": ["Gene_0", "Gene_1"],
            "beta_fate_mean": [0.5, -0.3],
            "posterior_inclusion_prob": [0.95, 0.80],
        })
        model._is_resolved = True

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "adata_prism.h5ad"))
            assert os.path.exists(os.path.join(tmpdir, "de_results.csv"))
            assert os.path.exists(os.path.join(tmpdir, "fate_probs.npy"))
            assert os.path.exists(os.path.join(tmpdir, "system_config.yaml"))

            # Load and verify
            loaded = PRISM.load(tmpdir)
            assert loaded.system_config.name == "skin"
            assert loaded._is_fitted
            assert loaded._is_resolved
            assert loaded._de_results is not None
            assert loaded._fate_probs is not None
            assert np.allclose(loaded._fate_probs, model._fate_probs)


class TestPRISMPlotMethods:
    def test_plot_embedding_raises_without_fit(self, small_adata):
        """plot_embedding raises if not fitted."""
        model = PRISM(small_adata, condition_key="genotype")
        with pytest.raises(RuntimeError, match="Must call fit"):
            model.plot_embedding()

    def test_plot_discriminators_raises_without_resolve(self, small_adata):
        """plot_discriminators raises if not resolved."""
        model = PRISM(small_adata, condition_key="genotype")
        with pytest.raises(RuntimeError, match="Must call resolve"):
            model.plot_discriminators()

    def test_get_discriminators_raises_without_resolve(self, small_adata):
        """get_discriminators raises if not resolved."""
        model = PRISM(small_adata, condition_key="genotype")
        with pytest.raises(RuntimeError, match="Must call resolve"):
            model.get_discriminators()

    def test_get_fate_probs_raises_without_resolve(self, small_adata):
        """get_fate_probs raises if not resolved."""
        model = PRISM(small_adata, condition_key="genotype")
        with pytest.raises(RuntimeError, match="Must call resolve"):
            model.get_fate_probs()
