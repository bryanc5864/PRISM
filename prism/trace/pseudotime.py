"""
PRISM-Trace: Perturbation-aware pseudotime inference.

Hybrid approach: DPT computed in PCA space (connected topology where
all cells are reachable) with branch assignment using PRISM-derived
fate probabilities (discriminative power).

Key insight: PRISM space *successfully* separates fate clusters, which
is ideal for classification but makes diffusion pseudotime fail (can't
cross between disconnected clusters). PCA space maintains connected
trajectories, enabling proper pseudotime ordering.
"""

import numpy as np
import scanpy as sc
import anndata as ad
from typing import Optional, Tuple, Dict
from scipy.stats import entropy as scipy_entropy, spearmanr


class PRISMPseudotime:
    """Hybrid PCA-pseudotime + PRISM-fate trajectory analysis.

    Combines:
    1. DPT in PCA space (connected topology, all cells reachable)
    2. PRISM fate probabilities (discriminative branch assignment)
    3. Temporal fate correlation analysis
    """

    def __init__(
        self,
        n_neighbors: int = 30,
        n_diffusion_components: int = 15,
    ):
        self.n_neighbors = n_neighbors
        self.n_dcs = n_diffusion_components

    def compute(
        self,
        adata: ad.AnnData,
        embedding_key: str = "X_pca",
        root_cluster: str = "Epi0",
        cluster_key: str = "cluster",
        genotype_key: str = "genotype",
        condition_branch_map: Optional[dict] = None,
    ) -> ad.AnnData:
        """Compute diffusion pseudotime in PCA space.

        Uses PCA space for DPT computation to ensure all cells are
        reachable via diffusion. Branch assignments are added using
        genotype labels as initial annotation.

        Args:
            adata: AnnData with PCA in obsm
            embedding_key: Key for embeddings (default "X_pca")
            root_cluster: Starting cluster for pseudotime
            cluster_key: Column with cluster labels
            genotype_key: Column with genotype labels

        Returns:
            adata with 'dpt_pseudotime' in obs
        """
        if embedding_key not in adata.obsm:
            if "X_pca" in adata.obsm:
                embedding_key = "X_pca"
            else:
                sc.pp.pca(adata, n_comps=50)
                embedding_key = "X_pca"

        neighbors_key = "pca_neighbors"
        sc.pp.neighbors(
            adata,
            use_rep=embedding_key,
            n_neighbors=self.n_neighbors,
            key_added=neighbors_key,
        )

        sc.tl.diffmap(adata, n_comps=self.n_dcs, neighbors_key=neighbors_key)

        # Set root cell
        if cluster_key in adata.obs.columns:
            root_mask = adata.obs[cluster_key] == root_cluster
            if root_mask.any():
                if "total_counts" in adata.obs:
                    root_idx = adata.obs.loc[root_mask, "total_counts"].idxmin()
                else:
                    root_idx = adata.obs.index[root_mask][0]
                adata.uns["iroot"] = np.where(adata.obs.index == root_idx)[0][0]
            else:
                adata.uns["iroot"] = 0
        else:
            adata.uns["iroot"] = 0

        sc.tl.dpt(adata, n_dcs=self.n_dcs, neighbors_key=neighbors_key)

        self._annotate_trajectories(adata, cluster_key, genotype_key, condition_branch_map)
        return adata

    def _annotate_trajectories(
        self,
        adata: ad.AnnData,
        cluster_key: str,
        genotype_key: str,
        condition_branch_map: Optional[dict] = None,
    ):
        """Annotate trajectory branches based on condition/genotype.

        Args:
            condition_branch_map: Maps condition names to branch names.
                Defaults to {"WT": "eccrine_branch", "En1-cKO": "hair_branch"}.
        """
        if "dpt_pseudotime" not in adata.obs:
            return

        if condition_branch_map is None:
            condition_branch_map = {"WT": "eccrine_branch", "En1-cKO": "hair_branch"}

        pseudotime = adata.obs["dpt_pseudotime"].values
        valid = np.isfinite(pseudotime)

        if not valid.any():
            return

        late_mask = valid & (pseudotime > np.percentile(pseudotime[valid], 70))
        adata.obs["trajectory_branch"] = "shared"

        if genotype_key in adata.obs.columns:
            for condition, branch_name in condition_branch_map.items():
                cond_late = late_mask & (adata.obs[genotype_key] == condition)
                if cond_late.any():
                    adata.obs.loc[cond_late, "trajectory_branch"] = branch_name

    def assign_fate_branches(
        self,
        adata: ad.AnnData,
        fate_probs: np.ndarray,
        percentile_threshold: float = 50,
        fate_names: Optional[list] = None,
        branch_names: Optional[dict] = None,
    ):
        """Assign branches using PRISM fate probabilities (more accurate).

        Cells in late pseudotime are assigned to branches based on
        which fate has highest probability. Supports N fates via argmax.

        Args:
            fate_names: List of fate names (e.g., ["uncommitted", "eccrine", "hair"])
            branch_names: Dict mapping branch_a/branch_b to names (for backward compat)
        """
        pseudotime = adata.obs["dpt_pseudotime"].values
        valid = np.isfinite(pseudotime)

        pt_thresh = np.percentile(pseudotime[valid], percentile_threshold)
        late_mask = valid & (pseudotime > pt_thresh)

        # Use argmax over non-uncommitted fates (columns 1..N)
        n_fates = fate_probs.shape[1]
        if n_fates <= 1:
            return np.zeros(len(adata), dtype=bool), np.zeros(len(adata), dtype=bool)

        # Best non-uncommitted fate for each cell
        best_fate = np.argmax(fate_probs[:, 1:], axis=1) + 1  # 1-indexed

        adata.obs["trajectory_branch"] = "shared"

        # Build branch masks for each non-uncommitted fate
        branch_masks = []
        if fate_names is None:
            fate_names = [f"fate_{i}" for i in range(n_fates)]

        for fate_idx in range(1, n_fates):
            mask = late_mask & (best_fate == fate_idx)
            fate_name = fate_names[fate_idx] if fate_idx < len(fate_names) else f"fate_{fate_idx}"
            branch_label = f"{fate_name}_branch"
            adata.obs.loc[mask, "trajectory_branch"] = branch_label
            branch_masks.append(mask)

        # Return first two branch masks for backward compat
        if len(branch_masks) >= 2:
            return branch_masks[0], branch_masks[1]
        elif len(branch_masks) == 1:
            return branch_masks[0], np.zeros(len(adata), dtype=bool)
        return np.zeros(len(adata), dtype=bool), np.zeros(len(adata), dtype=bool)

    def compute_branch_point(
        self,
        adata: ad.AnnData,
        fate_probs: np.ndarray,
        threshold: float = 0.3,
    ) -> Dict:
        """Identify the fate decision branch point.

        The branch point is where fate entropy drops below threshold,
        indicating commitment to a specific fate.
        """
        pseudotime = adata.obs["dpt_pseudotime"].values
        fate_entropy = scipy_entropy(fate_probs.T)

        valid = np.isfinite(pseudotime)
        pt_sorted_idx = np.argsort(pseudotime[valid])

        window = max(50, len(pt_sorted_idx) // 20)
        rolling_entropy = np.convolve(
            fate_entropy[valid][pt_sorted_idx],
            np.ones(window) / window,
            mode="valid"
        )

        below_threshold = np.where(rolling_entropy < threshold)[0]
        if len(below_threshold) > 0:
            branch_idx = below_threshold[0]
            branch_pseudotime = pseudotime[valid][pt_sorted_idx[branch_idx]]
            branch_ent = rolling_entropy[below_threshold[0]]
        else:
            branch_pseudotime = np.median(pseudotime[valid])
            branch_ent = np.mean(rolling_entropy)

        return {
            "branch_pseudotime": float(branch_pseudotime),
            "entropy_at_branch": float(branch_ent),
            "n_committed_cells": int(np.sum(fate_entropy < threshold)),
            "n_uncommitted_cells": int(np.sum(fate_entropy >= threshold)),
            "mean_entropy": float(np.mean(fate_entropy)),
        }

    def temporal_fate_correlation(
        self,
        adata: ad.AnnData,
        fate_probs: np.ndarray,
        gene_list: list,
        fdr_threshold: float = 0.05,
        fate_names: Optional[list] = None,
    ) -> "pd.DataFrame":
        """Find genes whose expression correlates with fate along pseudotime.

        For 2 fates: tests Spearman correlation between gene expression and
        (fate1_prob - fate2_prob). For N>2 fates: uses the first two non-uncommitted
        fates as the contrast axis.

        Args:
            fate_names: List of fate names for labeling directions.
        """
        import pandas as pd
        import scipy.sparse as sp

        if fate_names is None:
            fate_names = [f"fate_{i}" for i in range(fate_probs.shape[1])]

        pseudotime = adata.obs["dpt_pseudotime"].values
        valid = np.isfinite(pseudotime)

        # Build fate contrast score: first non-uncommitted fate vs second
        n_fates = fate_probs.shape[1]
        if n_fates >= 3:
            fate_score = fate_probs[:, 1] - fate_probs[:, 2]
            pos_label = f"{fate_names[1]}_corr" if len(fate_names) > 1 else "fate1_corr"
            neg_label = f"{fate_names[2]}_corr" if len(fate_names) > 2 else "fate2_corr"
        elif n_fates == 2:
            fate_score = fate_probs[:, 1]
            pos_label = f"{fate_names[1]}_corr" if len(fate_names) > 1 else "fate1_corr"
            neg_label = "uncommitted_corr"
        else:
            fate_score = fate_probs[:, 0]
            pos_label = "fate0_corr"
            neg_label = "neg_corr"

        pt_thresh = np.percentile(pseudotime[valid], 40)
        late_valid = valid & (pseudotime > pt_thresh)

        results = []
        for gene in gene_list:
            if gene not in adata.var_names:
                continue
            gene_idx = list(adata.var_names).index(gene)
            X = adata.X
            if sp.issparse(X):
                expr = X[:, gene_idx].toarray().flatten()
            else:
                expr = X[:, gene_idx].flatten()

            expr_late = expr[late_valid]
            fate_late = fate_score[late_valid]

            if len(expr_late) < 50:
                continue

            rho, p = spearmanr(expr_late, fate_late)
            results.append({
                "gene": gene,
                "spearman_rho": float(rho),
                "p_value": p,
                "direction": pos_label if rho > 0 else neg_label,
                "abs_rho": abs(rho),
            })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # BH correction
        pvals = df["p_value"].values
        n = len(pvals)
        sorted_idx = np.argsort(pvals)
        q = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            q[idx] = pvals[idx] * n / (i + 1)
        q_sorted = q[sorted_idx]
        for i in range(n - 2, -1, -1):
            q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
        q[sorted_idx] = q_sorted
        df["q_value"] = np.clip(q, 0, 1)

        sig = df[df["q_value"] < fdr_threshold]
        return sig.sort_values("q_value").reset_index(drop=True)
