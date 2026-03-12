"""
Branch-specific gene program analysis for PRISM-Trace.

At each branch point in the trajectory, identifies genes whose
expression changes significantly along one branch but not the other,
using GAM-based tests in PRISM embedding space.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from scipy.interpolate import UnivariateSpline
from scipy.stats import mannwhitneyu, pearsonr
import anndata as ad


class BranchAnalyzer:
    """Analyze branch-specific gene expression programs.

    Identifies gene cascades: temporal ordering of cryptic discriminators
    showing when each gene activates along the fate decision axis.
    """

    def __init__(
        self,
        n_splines: int = 5,
        fdr_threshold: float = 0.05,
    ):
        self.n_splines = n_splines
        self.fdr_threshold = fdr_threshold

    def find_branch_genes(
        self,
        adata: ad.AnnData,
        pseudotime_key: str = "dpt_pseudotime",
        branch_key: str = "trajectory_branch",
        branch_a: Optional[str] = None,
        branch_b: Optional[str] = None,
        gene_list: Optional[List[str]] = None,
        condition_key: str = "genotype",
        condition_branch_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Find genes differentially regulated between branches.

        For each gene, fits a GAM along pseudotime for each branch
        and tests whether the trajectories diverge.

        Args:
            adata: AnnData with pseudotime and branch annotations
            pseudotime_key: Column with pseudotime values
            branch_key: Column with branch assignments
            branch_a: Name of first branch (eccrine)
            branch_b: Name of second branch (hair)
            gene_list: Genes to test (None = all HVGs)

        Returns:
            DataFrame with gene, branch_diff_score, activation_time, direction
        """
        import scipy.sparse as sp

        # Default branch names
        if branch_a is None:
            branch_a = "eccrine_branch"
        if branch_b is None:
            branch_b = "hair_branch"
        if condition_branch_map is None:
            condition_branch_map = {"WT": "eccrine_branch", "En1-cKO": "hair_branch"}

        if gene_list is None:
            if "highly_variable" in adata.var:
                gene_list = adata.var_names[adata.var["highly_variable"]].tolist()
            else:
                gene_list = adata.var_names.tolist()

        pseudotime = adata.obs[pseudotime_key].values
        branch = adata.obs[branch_key].values

        mask_a = branch == branch_a
        mask_b = branch == branch_b

        if mask_a.sum() < 20 or mask_b.sum() < 20:
            print(f"Warning: Too few cells in branches ({mask_a.sum()}, {mask_b.sum()})")
            # Use all cells split by condition as fallback
            if condition_key in adata.obs:
                condition_names = sorted(condition_branch_map.keys())
                if len(condition_names) >= 2:
                    mask_a = adata.obs[condition_key] == condition_names[0]
                    mask_b = adata.obs[condition_key] == condition_names[1]

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

            result = self._test_branch_divergence(
                expr, pseudotime, mask_a, mask_b, gene,
                branch_a=branch_a, branch_b=branch_b,
            )
            if result is not None:
                results.append(result)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Multiple testing correction (BH)
        if "p_value" in df.columns:
            df = self._bh_correction(df)
            df = df[df["q_value"] < self.fdr_threshold]

        df = df.sort_values("branch_diff_score", ascending=False).reset_index(drop=True)

        return df

    def _test_branch_divergence(
        self,
        expr: np.ndarray,
        pseudotime: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        gene_name: str,
        branch_a: str = "eccrine_branch",
        branch_b: str = "hair_branch",
    ) -> Optional[Dict]:
        """Test whether a gene diverges between two branches."""
        valid_a = mask_a & np.isfinite(pseudotime)
        valid_b = mask_b & np.isfinite(pseudotime)

        if valid_a.sum() < 10 or valid_b.sum() < 10:
            return None

        pt_a = pseudotime[valid_a]
        expr_a = expr[valid_a]
        pt_b = pseudotime[valid_b]
        expr_b = expr[valid_b]

        try:
            # Fit splines along pseudotime for each branch
            sort_a = np.argsort(pt_a)
            sort_b = np.argsort(pt_b)

            # Use smoothing spline
            if len(pt_a[sort_a]) > self.n_splines + 1:
                spline_a = UnivariateSpline(
                    pt_a[sort_a], expr_a[sort_a],
                    k=min(3, self.n_splines), s=len(pt_a) * 0.5
                )
            else:
                return None

            if len(pt_b[sort_b]) > self.n_splines + 1:
                spline_b = UnivariateSpline(
                    pt_b[sort_b], expr_b[sort_b],
                    k=min(3, self.n_splines), s=len(pt_b) * 0.5
                )
            else:
                return None

            # Evaluate on common pseudotime grid
            pt_min = max(pt_a.min(), pt_b.min())
            pt_max = min(pt_a.max(), pt_b.max())

            if pt_max <= pt_min:
                return None

            grid = np.linspace(pt_min, pt_max, 100)
            pred_a = spline_a(grid)
            pred_b = spline_b(grid)

            # Branch divergence score: max absolute difference
            diff = np.abs(pred_a - pred_b)
            branch_diff_score = float(np.max(diff))

            # Activation time: pseudotime where difference first exceeds threshold
            threshold = 0.5 * branch_diff_score
            diverge_idx = np.where(diff > threshold)[0]
            activation_time = float(grid[diverge_idx[0]]) if len(diverge_idx) > 0 else float(grid[-1])

            # Direction: which branch has higher expression at max divergence
            max_diff_idx = np.argmax(diff)
            # Use branch names from parameters (strip _branch suffix if present)
            branch_a_label = branch_a.replace("_branch", "") if branch_a else "branch_a"
            branch_b_label = branch_b.replace("_branch", "") if branch_b else "branch_b"
            direction = f"{branch_a_label}_up" if pred_a[max_diff_idx] > pred_b[max_diff_idx] else f"{branch_b_label}_up"

            # Statistical test: Mann-Whitney on late pseudotime expression
            late_thresh = np.percentile(grid, 70)
            late_a = expr_a[pt_a > late_thresh] if (pt_a > late_thresh).any() else expr_a
            late_b = expr_b[pt_b > late_thresh] if (pt_b > late_thresh).any() else expr_b

            if len(late_a) > 3 and len(late_b) > 3:
                stat, p_value = mannwhitneyu(late_a, late_b, alternative="two-sided")
            else:
                p_value = 1.0

            return {
                "gene": gene_name,
                "branch_diff_score": branch_diff_score,
                "activation_time": activation_time,
                "direction": direction,
                "p_value": p_value,
                f"mean_expr_{branch_a_label}": float(expr_a.mean()),
                f"mean_expr_{branch_b_label}": float(expr_b.mean()),
                "log2fc": float(np.log2((expr_a.mean() + 1e-3) / (expr_b.mean() + 1e-3))),
            }

        except Exception:
            return None

    @staticmethod
    def _bh_correction(df: pd.DataFrame) -> pd.DataFrame:
        """Benjamini-Hochberg FDR correction."""
        from scipy.stats import false_discovery_control
        if "p_value" in df.columns and len(df) > 0:
            try:
                # scipy >= 1.11
                q_values = false_discovery_control(df["p_value"].values, method="bh")
                df["q_value"] = q_values
            except (ImportError, AttributeError):
                # Manual BH
                n = len(df)
                sorted_idx = np.argsort(df["p_value"].values)
                q_values = np.zeros(n)
                pvals = df["p_value"].values[sorted_idx]
                for i in range(n):
                    q_values[sorted_idx[i]] = pvals[i] * n / (i + 1)
                q_values = np.minimum.accumulate(q_values[::-1])[::-1]
                q_values = np.clip(q_values, 0, 1)
                df["q_value"] = q_values
        return df

    def build_gene_cascade(
        self,
        branch_df: pd.DataFrame,
        n_top: int = 30,
    ) -> pd.DataFrame:
        """Build temporal gene cascade from branch analysis.

        Orders genes by their activation time along the fate decision axis.

        Returns:
            DataFrame sorted by activation_time with cascade annotations
        """
        if branch_df.empty:
            return branch_df

        cascade = branch_df.nsmallest(n_top, "activation_time").copy()
        cascade["cascade_rank"] = range(1, len(cascade) + 1)

        # Categorize timing
        pt_range = cascade["activation_time"].max() - cascade["activation_time"].min()
        if pt_range > 0:
            normalized_time = (cascade["activation_time"] - cascade["activation_time"].min()) / pt_range
            cascade["timing_category"] = pd.cut(
                normalized_time,
                bins=3,
                labels=["early", "mid", "late"]
            )
        else:
            cascade["timing_category"] = "early"

        return cascade
