"""
Cross-species evolutionary analysis for PRISM-Trace.

Compares eccrine vs. hair fate decision dynamics between mouse and human
using ortholog-mapped gene sets and trajectory alignment.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import anndata as ad


class CrossSpeciesAnalyzer:
    """Cross-species comparison of fate decision trajectories.

    Identifies conserved vs. divergent cryptic discriminators between
    mouse and human, highlighting evolutionary innovations.
    """

    # Mouse-human ortholog mapping for key skin development genes
    ORTHOLOG_MAP = {
        # Eccrine markers
        "En1": "EN1", "Trpv6": "TRPV6", "Dkk4": "DKK4",
        "Foxi1": "FOXI1", "Defb6": "DEFB4A", "Lgr6": "LGR6",
        "Wif1": "WIF1", "Sfrp1": "SFRP1",
        # Hair markers
        "Lhx2": "LHX2", "Sox9": "SOX9", "Wnt10b": "WNT10B",
        "Shh": "SHH", "Edar": "EDAR", "Bmp4": "BMP4",
        "Lef1": "LEF1", "Ctnnb1": "CTNNB1",
        # EDEN/dermal markers
        "S100a4": "S100A4", "Pdgfra": "PDGFRA",
        "Col1a1": "COL1A1", "Col3a1": "COL3A1",
        "Dcn": "DCN", "Lum": "LUM",
        # Basal markers
        "Krt14": "KRT14", "Krt5": "KRT5", "Tp63": "TP63",
        "Itga6": "ITGA6", "Itgb1": "ITGB1",
        # Differentiation markers
        "Krt1": "KRT1", "Krt10": "KRT10",
    }

    def __init__(self):
        self.mouse_results = None
        self.human_results = None

    def map_orthologs(
        self,
        mouse_genes: List[str],
        custom_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Map mouse gene names to human orthologs.

        Args:
            mouse_genes: List of mouse gene names
            custom_map: Additional ortholog mappings

        Returns:
            Dict mapping mouse -> human gene names
        """
        ortholog_map = dict(self.ORTHOLOG_MAP)
        if custom_map:
            ortholog_map.update(custom_map)

        mapped = {}
        for gene in mouse_genes:
            if gene in ortholog_map:
                mapped[gene] = ortholog_map[gene]
            elif gene.upper() in [v.upper() for v in ortholog_map.values()]:
                # Already human name
                mapped[gene] = gene
            else:
                # Try case-insensitive match
                for m_gene, h_gene in ortholog_map.items():
                    if gene.lower() == m_gene.lower():
                        mapped[gene] = h_gene
                        break

        return mapped

    def compute_conservation_scores(
        self,
        mouse_discriminators: pd.DataFrame,
        human_discriminators: pd.DataFrame,
        ortholog_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Compute conservation scores for discriminator genes.

        A gene is "conserved" if it maintains high posterior inclusion
        probability in both species. "Divergent" genes are species-specific.

        Args:
            mouse_discriminators: PRISM-Resolve output for mouse
            human_discriminators: PRISM-Resolve output for human
            ortholog_map: Mouse -> human gene name mapping

        Returns:
            DataFrame with conservation scores for each gene
        """
        if ortholog_map is None:
            mouse_genes = mouse_discriminators["gene"].tolist()
            ortholog_map = self.map_orthologs(mouse_genes)

        results = []

        for _, row in mouse_discriminators.iterrows():
            mouse_gene = row["gene"]
            human_gene = ortholog_map.get(mouse_gene, None)

            mouse_pip = row.get("posterior_inclusion_prob", 0)
            mouse_beta = row.get("beta_fate_mean", 0)

            if human_gene and human_gene in human_discriminators["gene"].values:
                human_row = human_discriminators[
                    human_discriminators["gene"] == human_gene
                ].iloc[0]
                human_pip = human_row.get("posterior_inclusion_prob", 0)
                human_beta = human_row.get("beta_fate_mean", 0)

                # Conservation score: geometric mean of PIPs
                conservation = np.sqrt(mouse_pip * human_pip)

                # Direction concordance
                concordant = np.sign(mouse_beta) == np.sign(human_beta)

                category = "conserved" if (conservation > 0.5 and concordant) else \
                           "divergent_direction" if (conservation > 0.5 and not concordant) else \
                           "mouse_specific" if mouse_pip > 0.5 else "weak"
            else:
                human_pip = 0
                human_beta = 0
                conservation = 0
                concordant = False
                category = "mouse_specific" if mouse_pip > 0.5 else "unmapped"

            results.append({
                "mouse_gene": mouse_gene,
                "human_gene": human_gene or "unmapped",
                "mouse_pip": mouse_pip,
                "human_pip": human_pip,
                "mouse_beta": mouse_beta,
                "human_beta": human_beta,
                "conservation_score": conservation,
                "direction_concordant": concordant,
                "category": category,
            })

        df = pd.DataFrame(results)
        df = df.sort_values("conservation_score", ascending=False).reset_index(drop=True)

        return df

    def align_trajectories(
        self,
        mouse_adata: ad.AnnData,
        human_adata: ad.AnnData,
        ortholog_map: Optional[Dict[str, str]] = None,
        pseudotime_key: str = "dpt_pseudotime",
        embedding_key: str = "X_prism",
    ) -> Dict:
        """Align developmental trajectories between species.

        Uses dynamic time warping (DTW) on ortholog-mapped gene
        expression along pseudotime to find corresponding stages.

        Returns:
            Dict with alignment scores and corresponding stages
        """
        if ortholog_map is None:
            ortholog_map = self.map_orthologs(mouse_adata.var_names.tolist())

        # Find common orthologs
        common_mouse = [g for g in ortholog_map.keys() if g in mouse_adata.var_names]
        common_human = [ortholog_map[g] for g in common_mouse if ortholog_map[g] in human_adata.var_names]
        common_mouse = [g for g in common_mouse if ortholog_map[g] in human_adata.var_names]

        if len(common_mouse) < 5:
            return {"error": "Too few common orthologs", "n_common": len(common_mouse)}

        # Get pseudotime-ordered expression
        mouse_pt = mouse_adata.obs[pseudotime_key].values
        human_pt = human_adata.obs[pseudotime_key].values

        mouse_valid = np.isfinite(mouse_pt)
        human_valid = np.isfinite(human_pt)

        # Bin expression by pseudotime
        n_bins = 50
        mouse_binned = self._bin_by_pseudotime(
            mouse_adata[mouse_valid, common_mouse], mouse_pt[mouse_valid], n_bins
        )
        human_binned = self._bin_by_pseudotime(
            human_adata[human_valid, common_human], human_pt[human_valid], n_bins
        )

        # Compute correlation between binned trajectories
        from scipy.stats import spearmanr
        correlations = []
        for i in range(len(common_mouse)):
            corr, _ = spearmanr(mouse_binned[:, i], human_binned[:, i])
            correlations.append(corr if np.isfinite(corr) else 0)

        return {
            "n_common_orthologs": len(common_mouse),
            "mean_trajectory_correlation": float(np.mean(correlations)),
            "median_trajectory_correlation": float(np.median(correlations)),
            "per_gene_correlations": dict(zip(common_mouse, [float(c) for c in correlations])),
            "highly_conserved_genes": [
                common_mouse[i] for i, c in enumerate(correlations) if c > 0.7
            ],
            "divergent_genes": [
                common_mouse[i] for i, c in enumerate(correlations) if c < 0.3
            ],
        }

    @staticmethod
    def _bin_by_pseudotime(
        adata: ad.AnnData,
        pseudotime: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        """Bin gene expression by pseudotime."""
        import scipy.sparse as sp

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()

        bins = np.linspace(pseudotime.min(), pseudotime.max(), n_bins + 1)
        binned = np.zeros((n_bins, X.shape[1]))

        for i in range(n_bins):
            mask = (pseudotime >= bins[i]) & (pseudotime < bins[i + 1])
            if mask.sum() > 0:
                binned[i] = X[mask].mean(axis=0)

        return binned
