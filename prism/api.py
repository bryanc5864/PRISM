"""
PRISM High-Level API.

Provides a simple interface for applying PRISM to any scRNA-seq dataset
with perturbation-induced cryptic cell fate decisions.

Usage:
    import prism
    model = prism.PRISM(adata, condition_key="genotype")
    model.preprocess()
    model.fit(n_epochs=50)
    model.resolve(method="fast")
    model.trace()
    discriminators = model.get_discriminators(pip_threshold=0.5)
"""

import os
import pickle
import numpy as np
import anndata as ad
from typing import Optional, Dict, List, Any

from .config import SystemConfig, SKIN_CONFIG


class PRISM:
    """High-level PRISM interface for cryptic cell fate analysis.

    Wraps the full PRISM pipeline (Encode, Resolve, Trace) in a
    simple API that works with any biological system.

    Args:
        adata: AnnData object with raw or normalized counts
        condition_key: obs column identifying perturbation conditions
        config: Optional SystemConfig for the biological system
        system: Optional path to system YAML config file
    """

    def __init__(
        self,
        adata: ad.AnnData,
        condition_key: str = "genotype",
        config: Optional[SystemConfig] = None,
        system: Optional[str] = None,
    ):
        self.adata = adata
        self.condition_key = condition_key

        # Load system config
        if config is not None:
            self.system_config = config
        elif system is not None:
            self.system_config = SystemConfig.from_yaml(system)
        else:
            self.system_config = SystemConfig(
                name="custom",
                condition_key=condition_key,
                conditions={},
                fate_names=["uncommitted", "fate_a", "fate_b"],
            )

        # Internal state
        self._encoder = None
        self._trainer = None
        self._de_results = None
        self._fate_probs = None
        self._is_preprocessed = False
        self._is_fitted = False
        self._is_resolved = False
        self._is_traced = False

        # Training config (defaults)
        self._train_config = {
            "d_model": 512,
            "n_layers": 12,
            "n_heads": 8,
            "d_ff": 1024,
            "d_output": 256,
            "n_expression_bins": 51,
            "dropout": 0.1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "projection_dims": [512, 256, 128],
            "temperature_init": 0.07,
            "alpha_max": 2.0,
            "curriculum_warmup_epochs": 10,
            "lr_lora": 2e-4,
            "lr_head": 1e-3,
            "weight_decay": 0.01,
            "batch_size": 256,
            "info_reg_lambda": 0.1,
            "recon_weight": 0.1,
            "scheduler": "cosine",
            "gradient_clip": 1.0,
            "seed": 42,
            "device": "cuda:0",
        }

    def preprocess(
        self,
        min_genes: int = 200,
        max_genes: int = 5000,
        max_mito_pct: float = 5.0,
        n_hvgs: int = 2000,
        **kwargs,
    ) -> "PRISM":
        """Preprocess the AnnData object.

        Performs QC filtering, normalization, HVG selection, PCA,
        condition assignment, and fate label assignment.
        """
        from .data.preprocess import (
            preprocess_adata,
            assign_genotypes,
            assign_labels,
            compute_harmony_baseline,
        )

        forced_genes = self.system_config.forced_genes or []

        self.adata = preprocess_adata(
            self.adata,
            min_genes=min_genes,
            max_genes=max_genes,
            max_mito_pct=max_mito_pct,
            n_hvgs=n_hvgs,
            forced_genes=forced_genes,
            **kwargs,
        )

        # Assign conditions
        self.adata = assign_genotypes(
            self.adata,
            sample_condition_map=self.system_config.sample_condition_map or None,
            condition_key=self.system_config.condition_key,
        )

        # Assign fate labels
        self.adata = assign_labels(
            self.adata,
            condition_key=self.system_config.condition_key,
            conditions=self.system_config.conditions,
            marker_scores=self.system_config.marker_scores or None,
            fate_categories=self.system_config.fate_categories,
            label_strategy=self.system_config.label_strategy,
            annotation_key=self.system_config.annotation_key,
            annotation_fate_map=self.system_config.annotation_fate_map or None,
        )

        # Harmony baseline
        self.adata = compute_harmony_baseline(self.adata)

        self._is_preprocessed = True
        return self

    def fit(
        self,
        n_epochs: int = 50,
        patience: int = 10,
        device: str = "cuda:0",
        auto_hard_negatives: bool = True,
        **kwargs,
    ) -> "PRISM":
        """Train PRISM-Encode.

        Args:
            n_epochs: Number of training epochs
            patience: Early stopping patience
            device: CUDA device
            auto_hard_negatives: Auto-construct hard negatives from cross-condition neighbors
            **kwargs: Override any training config parameter
        """
        import torch
        from .data.dataset import PRISMDataset, build_dataloaders
        from .data.preprocess import split_data
        from .models.encoder import PRISMEncoder
        from .training.trainer import PRISMTrainer
        from torch.utils.data import DataLoader

        config = {**self._train_config, "device": device, **kwargs}
        config["n_fate_categories"] = len(self.system_config.fate_categories)
        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        condition_key = self.system_config.condition_key

        # Split data
        train_adata, val_adata, test_adata = split_data(self.adata, seed=seed, condition_key=condition_key)

        # Create datasets
        n_genes = min(
            config.get("n_genes", 2000),
            self.adata.var["highly_variable"].sum() if "highly_variable" in self.adata.var else 2000,
        )

        train_dataset = PRISMDataset(train_adata, n_genes=n_genes, condition_key=condition_key)
        val_dataset = PRISMDataset(val_adata, n_genes=n_genes, condition_key=condition_key)

        batch_size = config["batch_size"]
        train_loader, val_loader = build_dataloaders(
            train_dataset, val_dataset,
            batch_size=batch_size, num_workers=0, seed=seed,
        )

        # Build encoder
        self._encoder = PRISMEncoder(
            n_genes=n_genes,
            n_bins=config.get("n_expression_bins", 51),
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            d_output=config["d_output"],
            dropout=config["dropout"],
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            projection_dims=config["projection_dims"],
        )

        # Train
        self._trainer = PRISMTrainer(self._encoder, config, device=device)
        self._trainer.train(
            train_loader, val_loader,
            n_epochs=n_epochs,
            patience=patience,
        )

        # Auto hard-negative construction
        if auto_hard_negatives and self.system_config.condition_key in self.adata.obs:
            self._build_hard_negatives()

        # Extract embeddings for all cells
        full_dataset = PRISMDataset(self.adata, n_genes=n_genes, condition_key=condition_key)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        full_embeddings, _, _ = self._trainer.extract_embeddings(full_loader)
        self.adata.obsm["X_prism"] = full_embeddings

        self._is_fitted = True
        return self

    def _build_hard_negatives(self):
        """Auto-construct hard negatives from cross-condition PCA nearest neighbors.

        For each cell, finds its nearest neighbors in PCA space that belong to
        a different condition. These transcriptionally similar but condition-divergent
        pairs are the hardest negatives for the contrastive loss.
        """
        from sklearn.neighbors import NearestNeighbors

        if "X_pca" not in self.adata.obsm:
            return

        condition_key = self.system_config.condition_key
        if condition_key not in self.adata.obs:
            return

        pca = self.adata.obsm["X_pca"][:, :30]
        conditions = self.adata.obs[condition_key].values

        # Find k nearest neighbors in PCA space
        k = 20
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(pca)
        distances, indices = nn.kneighbors(pca)

        # For each cell, identify cross-condition neighbors
        cross_condition_pairs = []
        for i in range(len(conditions)):
            for j_idx in range(1, k + 1):  # skip self
                j = indices[i, j_idx]
                if conditions[i] != conditions[j]:
                    cross_condition_pairs.append((i, j, distances[i, j_idx]))

        # Store as sparse similarity matrix in adata.obsp
        if cross_condition_pairs:
            import scipy.sparse as sp
            n = len(conditions)
            rows = [p[0] for p in cross_condition_pairs]
            cols = [p[1] for p in cross_condition_pairs]
            vals = [1.0 - p[2] for p in cross_condition_pairs]  # convert distance to similarity
            hard_neg_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
            self.adata.obsp["hard_negative_sim"] = hard_neg_matrix
            print(f"Auto hard-negatives: {len(cross_condition_pairs)} cross-condition pairs")

    def resolve(
        self,
        method: str = "fast",
        pip_threshold: float = 0.5,
        **kwargs,
    ) -> "PRISM":
        """Run PRISM-Resolve: fate assignment + horseshoe DE.

        Args:
            method: "fast" (BayesianRidge), "mcmc" (reduced NUTS), or "full" (full NUTS)
            pip_threshold: PIP threshold for reporting discriminators
        """
        from .resolve.mixture import BayesianFateMixture
        from .resolve.horseshoe import HorseshoeDE
        import scipy.sparse as sp

        if "X_prism" not in self.adata.obsm:
            raise RuntimeError("Must call fit() before resolve()")

        embeddings = self.adata.obsm["X_prism"]
        fate_names = self.system_config.fate_names
        known_threshold = self.system_config.known_fate_threshold

        # Fate assignment
        mixture = BayesianFateMixture(
            n_components=len(fate_names),
            fate_names=fate_names,
        )

        labels = self.adata.obs["fate_int"].values if "fate_int" in self.adata.obs else None
        label_mask = labels >= known_threshold if labels is not None else None

        mixture.fit(embeddings, labels, label_mask)
        self._fate_probs = mixture.predict_proba(embeddings)

        for i, name in enumerate(fate_names):
            if i < self._fate_probs.shape[1]:
                self.adata.obs[f"prism_{name}_prob"] = self._fate_probs[:, i]

        # Horseshoe DE
        hvg_mask = self.adata.var["highly_variable"] if "highly_variable" in self.adata.var else np.ones(self.adata.shape[1], dtype=bool)
        gene_names = self.adata.var_names[hvg_mask].tolist()

        X = self.adata[:, hvg_mask].X
        if sp.issparse(X):
            X = X.toarray()

        de = HorseshoeDE(**kwargs)
        fate_prob_col = self._fate_probs[:, 1] if self._fate_probs.shape[1] > 1 else np.random.rand(len(self.adata))

        if method == "mcmc":
            self._de_results = de.fit_mcmc(X, fate_prob_col, gene_names)
        elif method == "full":
            self._de_results = de.fit(X, fate_prob_col, gene_names)
        else:
            self._de_results = de.fit_fast(X, fate_prob_col, gene_names)

        n_sig = (self._de_results["posterior_inclusion_prob"] > pip_threshold).sum()
        print(f"PRISM-Resolve: {n_sig} genes with PIP > {pip_threshold}")

        self._is_resolved = True
        return self

    def trace(self, **kwargs) -> "PRISM":
        """Run PRISM-Trace: pseudotime + branch analysis."""
        from .trace.pseudotime import PRISMPseudotime

        if self._fate_probs is None:
            raise RuntimeError("Must call resolve() before trace()")

        pt = PRISMPseudotime(
            n_neighbors=kwargs.get("n_neighbors", 30),
            n_diffusion_components=kwargs.get("n_diffusion_components", 15),
        )

        self.adata = pt.compute(
            self.adata,
            embedding_key="X_pca",
            root_cluster=self.system_config.root_cluster,
            cluster_key=self.system_config.cluster_key,
            genotype_key=self.system_config.condition_key,
            condition_branch_map=self.system_config.condition_branch_map,
        )

        fate_names = self.system_config.fate_names
        branch_names = self.system_config.branch_names
        pt.assign_fate_branches(
            self.adata, self._fate_probs, percentile_threshold=50,
            fate_names=fate_names, branch_names=branch_names,
        )

        # Temporal fate correlation
        hvg_mask = self.adata.var["highly_variable"] if "highly_variable" in self.adata.var else np.ones(self.adata.shape[1], dtype=bool)
        gene_list = self.adata.var_names[hvg_mask].tolist()
        corr_df = pt.temporal_fate_correlation(self.adata, self._fate_probs, gene_list, fate_names=fate_names)

        self._corr_df = corr_df
        self._is_traced = True
        return self

    def plot_embedding(
        self,
        color_by: str = "fate",
        save_path: Optional[str] = None,
        **kwargs,
    ):
        """Plot UMAP of PRISM embedding colored by fate or other metadata.

        Args:
            color_by: "fate" for fate labels, or any obs column name
            save_path: optional path to save figure (PNG)
            **kwargs: passed to plot_umap_comparison
        """
        from .utils.visualization import plot_umap_comparison

        if "X_prism" not in self.adata.obsm:
            raise RuntimeError("Must call fit() before plot_embedding()")

        embeddings_dict = {"PRISM": self.adata.obsm["X_prism"]}
        if "X_pca" in self.adata.obsm:
            embeddings_dict = {"PCA": self.adata.obsm["X_pca"][:, :30], **embeddings_dict}
        if "X_harmony" in self.adata.obsm:
            embeddings_dict["Harmony"] = self.adata.obsm["X_harmony"][:, :30]
            # Reorder: PCA, Harmony, PRISM
            embeddings_dict = {k: embeddings_dict[k] for k in ["PCA", "Harmony", "PRISM"] if k in embeddings_dict}

        if color_by == "fate":
            labels = self.adata.obs["fate_int"].values if "fate_int" in self.adata.obs else np.zeros(len(self.adata))
        else:
            labels = self.adata.obs[color_by].astype("category").cat.codes.values if color_by in self.adata.obs.columns else np.zeros(len(self.adata))

        plot_umap_comparison(
            embeddings_dict, labels,
            save_path=save_path or "figures/prism_embedding.png",
            **kwargs,
        )

    def plot_discriminators(
        self,
        n_top: int = 20,
        save_path: Optional[str] = None,
    ):
        """Plot top discriminator genes from PRISM-Resolve.

        Args:
            n_top: number of top genes to show
            save_path: optional path to save figure (PNG)
        """
        from .utils.visualization import plot_discriminator_genes

        if self._de_results is None:
            raise RuntimeError("Must call resolve() before plot_discriminators()")

        plot_discriminator_genes(
            self._de_results, n_top=n_top,
            save_path=save_path or "figures/discriminator_genes.png",
        )

    def get_discriminators(self, pip_threshold: float = 0.5) -> "pd.DataFrame":
        """Get discriminator genes above PIP threshold.

        Returns:
            DataFrame with gene names, effect sizes, and PIPs
        """
        if self._de_results is None:
            raise RuntimeError("Must call resolve() first")

        return self._de_results[
            self._de_results["posterior_inclusion_prob"] > pip_threshold
        ].reset_index(drop=True)

    def get_fate_probs(self) -> np.ndarray:
        """Get fate probability matrix (N cells x K fates)."""
        if self._fate_probs is None:
            raise RuntimeError("Must call resolve() first")
        return self._fate_probs

    def save(self, path: str):
        """Save PRISM results to disk.

        Saves the AnnData (with embeddings) and DE results.
        """
        os.makedirs(path, exist_ok=True)

        # Save adata
        self.adata.write_h5ad(os.path.join(path, "adata_prism.h5ad"))

        # Save DE results
        if self._de_results is not None:
            self._de_results.to_csv(os.path.join(path, "de_results.csv"), index=False)

        # Save fate probs
        if self._fate_probs is not None:
            np.save(os.path.join(path, "fate_probs.npy"), self._fate_probs)

        # Save system config
        self.system_config.to_yaml(os.path.join(path, "system_config.yaml"))

        # Save model checkpoint
        if self._trainer is not None:
            import torch
            torch.save(
                self._encoder.state_dict(),
                os.path.join(path, "encoder.pt"),
            )

        print(f"Saved PRISM results to {path}")

    def cross_species(
        self,
        human_adata: ad.AnnData,
        human_de_results: Optional["pd.DataFrame"] = None,
        ortholog_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Compare fate decisions across species.

        Wraps CrossSpeciesAnalyzer for conservation scoring and
        trajectory alignment between mouse and human data.

        Args:
            human_adata: Human AnnData with PRISM embeddings and pseudotime
            human_de_results: Human PRISM-Resolve DE results (optional)
            ortholog_map: Custom mouse→human gene mapping (auto-generated if None)

        Returns:
            Dict with 'conservation' and/or 'trajectory_alignment' results
        """
        from .trace.evolution import CrossSpeciesAnalyzer

        analyzer = CrossSpeciesAnalyzer()
        if ortholog_map is None:
            ortholog_map = analyzer.map_orthologs(self.adata.var_names.tolist())

        results = {}

        if self._de_results is not None and human_de_results is not None:
            results["conservation"] = analyzer.compute_conservation_scores(
                self._de_results, human_de_results, ortholog_map
            )

        if "dpt_pseudotime" in self.adata.obs and "dpt_pseudotime" in human_adata.obs:
            results["trajectory_alignment"] = analyzer.align_trajectories(
                self.adata, human_adata, ortholog_map
            )

        return results

    @classmethod
    def load(cls, path: str) -> "PRISM":
        """Load PRISM results from disk."""
        import pandas as pd

        adata = ad.read_h5ad(os.path.join(path, "adata_prism.h5ad"))
        system_config = SystemConfig.from_yaml(os.path.join(path, "system_config.yaml"))

        model = cls(adata, condition_key=system_config.condition_key, config=system_config)
        model._is_preprocessed = True
        model._is_fitted = "X_prism" in adata.obsm

        de_path = os.path.join(path, "de_results.csv")
        if os.path.exists(de_path):
            model._de_results = pd.read_csv(de_path)
            model._is_resolved = True

        fp_path = os.path.join(path, "fate_probs.npy")
        if os.path.exists(fp_path):
            model._fate_probs = np.load(fp_path)

        print(f"Loaded PRISM results from {path}")
        return model
