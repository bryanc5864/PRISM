"""Tests for SystemConfig loading, saving, and roundtrip."""

import os
import tempfile
import pytest
from prism.config import SystemConfig, SKIN_CONFIG


class TestSystemConfig:
    def test_skin_config_defaults(self):
        """SKIN_CONFIG has expected fields."""
        assert SKIN_CONFIG.name == "skin"
        assert SKIN_CONFIG.condition_key == "genotype"
        assert "WT" in SKIN_CONFIG.conditions
        assert len(SKIN_CONFIG.fate_names) == 3
        assert SKIN_CONFIG.label_strategy == "hierarchical"

    def test_from_yaml_all_systems(self):
        """All 4 system YAML configs load without error."""
        configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        for name in ["skin", "pancreas", "cortex", "hsc"]:
            path = os.path.join(configs_dir, f"{name}.yaml")
            if os.path.exists(path):
                cfg = SystemConfig.from_yaml(path)
                assert cfg.name == name
                assert len(cfg.fate_names) >= 2
                assert len(cfg.fate_categories) >= 3
                assert cfg.condition_key

    def test_to_yaml_roundtrip(self, skin_config):
        """Config survives save/load roundtrip."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            skin_config.to_yaml(path)
            loaded = SystemConfig.from_yaml(path)

            assert loaded.name == skin_config.name
            assert loaded.condition_key == skin_config.condition_key
            assert loaded.conditions == skin_config.conditions
            assert loaded.fate_names == skin_config.fate_names
            assert loaded.label_strategy == skin_config.label_strategy
            assert loaded.fate_categories == skin_config.fate_categories
            assert loaded.forced_genes == skin_config.forced_genes
        finally:
            os.unlink(path)

    def test_annotation_fields(self):
        """HSC config has annotation fields when loaded."""
        configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        hsc_path = os.path.join(configs_dir, "hsc.yaml")
        if os.path.exists(hsc_path):
            cfg = SystemConfig.from_yaml(hsc_path)
            assert cfg.label_strategy == "annotation"
            assert cfg.annotation_key == "cell_type_annotation"
            assert len(cfg.annotation_fate_map) > 0
            assert "Ery" in cfg.annotation_fate_map

    def test_default_annotation_fields(self):
        """Default SystemConfig has empty annotation fields."""
        cfg = SystemConfig(
            name="test",
            condition_key="genotype",
            conditions={"WT": 0},
            fate_names=["a", "b"],
        )
        assert cfg.annotation_key == ""
        assert cfg.annotation_fate_map == {}
