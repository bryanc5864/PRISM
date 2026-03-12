"""Tests for PRISMDataset and ContrastiveSampler."""

import numpy as np
import torch
import pytest

from prism.data.dataset import PRISMDataset, ContrastiveSampler, build_dataloaders


class TestPRISMDataset:
    def test_getitem_shapes(self, small_adata):
        """__getitem__ returns correct tensor shapes."""
        ds = PRISMDataset(small_adata, n_genes=500, condition_key="genotype")

        item = ds[0]
        assert "expression" in item
        assert "raw_expression" in item
        assert "genotype" in item
        assert "fate_label" in item
        assert "cell_idx" in item

        assert item["expression"].shape == (500,)
        assert item["raw_expression"].shape == (500,)
        assert item["expression"].dtype == torch.long
        assert item["raw_expression"].dtype == torch.float32

    def test_len(self, small_adata):
        """Dataset length matches adata."""
        ds = PRISMDataset(small_adata, n_genes=500, condition_key="genotype")
        assert len(ds) == small_adata.shape[0]

    def test_rank_encoding(self, small_adata):
        """Rank-value encoding produces valid bin indices."""
        ds = PRISMDataset(small_adata, n_genes=500, n_bins=51, condition_key="genotype")

        item = ds[0]
        expr = item["expression"].numpy()
        # All values should be in [0, n_bins-1]
        assert expr.min() >= 0
        assert expr.max() <= 50

    def test_condition_encoding(self, small_adata):
        """Conditions are encoded as integers."""
        ds = PRISMDataset(small_adata, n_genes=500, condition_key="genotype")

        # Check that genotype values are 0 or 1
        all_geno = set()
        for i in range(len(ds)):
            all_geno.add(ds[i]["genotype"].item())
        assert all_geno == {0, 1}

    def test_single_condition(self, small_adata):
        """Dataset handles single condition gracefully."""
        small_adata.obs["genotype"] = "WT"
        ds = PRISMDataset(small_adata, n_genes=500, condition_key="genotype")
        item = ds[0]
        assert item["genotype"].item() == 0

    def test_n_genes_truncation(self, small_adata):
        """Dataset correctly truncates to n_genes."""
        ds = PRISMDataset(small_adata, n_genes=100, condition_key="genotype")
        item = ds[0]
        assert item["expression"].shape == (100,)
        assert item["raw_expression"].shape == (100,)


class TestContrastiveSampler:
    def test_batch_size(self, small_adata):
        """Sampler produces batches of correct size."""
        ds = PRISMDataset(small_adata, n_genes=500, condition_key="genotype")
        sampler = ContrastiveSampler(ds, batch_size=16, seed=42)

        for batch in sampler:
            assert len(batch) == 16
            break

    def test_balancing(self, small_adata):
        """Balanced sampling produces roughly equal WT/KO counts."""
        ds = PRISMDataset(small_adata, n_genes=500, condition_key="genotype")
        sampler = ContrastiveSampler(ds, batch_size=32, balance_genotype=True, seed=42)

        wt_count = 0
        ko_count = 0
        for batch in sampler:
            for idx in batch:
                if ds.genotype[idx] == 0:
                    wt_count += 1
                else:
                    ko_count += 1
            break

        # Should be roughly balanced (16 each)
        assert abs(wt_count - ko_count) <= 2

    def test_single_condition_disables_balancing(self, small_adata):
        """Sampler auto-disables balancing with single condition."""
        small_adata.obs["genotype"] = "WT"
        ds = PRISMDataset(small_adata, n_genes=500, condition_key="genotype")
        sampler = ContrastiveSampler(ds, batch_size=16, balance_genotype=True, seed=42)

        assert sampler.balance_genotype is False
        # Should still produce batches
        for batch in sampler:
            assert len(batch) == 16
            break


class TestBuildDataloaders:
    def test_creates_loaders(self, small_adata):
        """build_dataloaders returns two DataLoaders."""
        # Split the data manually
        train_adata = small_adata[:80].copy()
        val_adata = small_adata[80:].copy()

        train_ds = PRISMDataset(train_adata, n_genes=500, condition_key="genotype")
        val_ds = PRISMDataset(val_adata, n_genes=500, condition_key="genotype")

        train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=16, num_workers=0, seed=42)

        # Should be iterable
        batch = next(iter(train_loader))
        assert "expression" in batch
        assert batch["expression"].shape[0] == 16
