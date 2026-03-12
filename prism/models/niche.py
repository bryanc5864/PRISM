"""
Niche Context Module for PRISM.

Integrates dermal niche information into epidermal cell embeddings
using CellChat-derived receptor-ligand signaling scores from
Dingwall et al. (2024, Data S2).

For each epidermal cell i:
n_i = Σ_j L-R_score(i,j) · z_j^derm

where L-R_score is the CellChat signaling probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


class NicheContextModule(nn.Module):
    """Compute niche context vectors from receptor-ligand signaling.

    Rather than requiring spatial data, this module uses computationally
    derived signaling probabilities from CellChat to weight dermal
    cell contributions to each epidermal cell's niche context.
    """

    def __init__(
        self,
        d_derm: int = 512,      # Dermal cell embedding dimension
        d_niche: int = 64,      # Output niche context dimension
        n_lr_pairs: int = 50,   # Number of ligand-receptor pairs
        d_signal: int = 32,     # Signal embedding dimension
    ):
        super().__init__()

        # Project dermal embeddings
        self.derm_projection = nn.Sequential(
            nn.Linear(d_derm, 128),
            nn.ReLU(),
            nn.Linear(128, d_niche),
        )

        # Signaling pair embeddings
        self.lr_embedding = nn.Embedding(n_lr_pairs, d_signal)

        # Attention for weighting different L-R pair contributions
        self.signal_attention = nn.Sequential(
            nn.Linear(d_signal + d_niche, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Final projection
        self.output_projection = nn.Linear(d_niche, d_niche)

    def forward(
        self,
        derm_embeddings: torch.Tensor,     # (N_derm, d_derm)
        lr_scores: torch.Tensor,           # (N_epi, N_derm, n_lr_pairs) signaling probabilities
        lr_pair_ids: Optional[torch.Tensor] = None,  # (n_lr_pairs,) indices
    ) -> torch.Tensor:
        """Compute niche context for epidermal cells.

        Args:
            derm_embeddings: Embeddings of dermal cells from the encoder
            lr_scores: CellChat-derived ligand-receptor signaling probabilities
            lr_pair_ids: Indices of active L-R pairs

        Returns:
            niche_context: (N_epi, d_niche) niche context vectors
        """
        N_epi = lr_scores.shape[0]
        N_derm = derm_embeddings.shape[0]

        # Project dermal embeddings
        derm_proj = self.derm_projection(derm_embeddings)  # (N_derm, d_niche)

        # Weighted sum over dermal cells for each epidermal cell
        # Sum over L-R pairs first
        total_lr_score = lr_scores.sum(dim=-1)  # (N_epi, N_derm)
        total_lr_score = F.softmax(total_lr_score, dim=-1)  # normalize

        # Weighted dermal contribution
        niche_context = torch.matmul(total_lr_score, derm_proj)  # (N_epi, d_niche)

        return self.output_projection(niche_context)

    def compute_from_precomputed(
        self,
        niche_features: torch.Tensor,  # (N_epi, d_niche) precomputed
    ) -> torch.Tensor:
        """Use precomputed niche features (e.g., from CellChat analysis).

        For efficiency, niche features can be precomputed once and stored.
        """
        return self.output_projection(niche_features)


def precompute_niche_features(
    adata,
    epi_mask: np.ndarray,
    derm_mask: np.ndarray,
    lr_pairs: Optional[Dict] = None,
    d_niche: int = 64,
) -> np.ndarray:
    """Precompute niche context features from CellChat results.

    This function computes a simplified niche context vector for each
    epidermal cell based on its proximity to dermal cells and known
    L-R signaling pairs.

    In practice, this uses the CellChat output from Dingwall et al.
    (Data S2) which provides signaling probabilities between
    epidermal and dermal subclusters.

    Args:
        adata: AnnData with both epidermal and dermal cells
        epi_mask: Boolean mask for epidermal cells
        derm_mask: Boolean mask for dermal cells
        lr_pairs: Dict of known L-R pairs with signaling scores
        d_niche: Dimension of niche feature vector

    Returns:
        niche_features: (N_epi, d_niche) array of niche context features
    """
    import scipy.sparse as sp

    N_epi = epi_mask.sum()
    N_derm = derm_mask.sum()

    if lr_pairs is None:
        # Use expression-based proxy for niche signaling
        # Key EDEN markers from Dingwall et al.
        eden_markers = ["S100a4", "Pdgfra", "Col1a1", "Dcn", "Lum", "Col3a1"]
        eccrine_receptors = ["Lgr6", "Trpv6", "Edar"]

        available_markers = [g for g in eden_markers if g in adata.var_names]
        available_receptors = [g for g in eccrine_receptors if g in adata.var_names]

        if not available_markers or not available_receptors:
            # Return zero features if markers not available
            return np.zeros((N_epi, d_niche), dtype=np.float32)

        # Dermal ligand expression
        X_derm = adata[derm_mask, available_markers].X
        if sp.issparse(X_derm):
            X_derm = X_derm.toarray()

        # Epidermal receptor expression
        X_epi = adata[epi_mask, available_receptors].X
        if sp.issparse(X_epi):
            X_epi = X_epi.toarray()

        # Compute approximate L-R signaling scores
        # Using outer product of mean ligand/receptor expression per cluster
        derm_mean = X_derm.mean(axis=0)  # average ligand expression
        epi_vals = X_epi  # per-cell receptor expression

        # Niche feature = receptor expression * mean ligand availability
        # Pad or truncate to d_niche
        raw_niche = np.zeros((N_epi, d_niche), dtype=np.float32)
        n_features = min(len(available_receptors), d_niche)
        raw_niche[:, :n_features] = epi_vals[:, :n_features] * derm_mean[:n_features].reshape(1, -1)

        return raw_niche

    else:
        # Use provided CellChat L-R pair scores
        niche_features = np.zeros((N_epi, d_niche), dtype=np.float32)
        for i, (lr_name, scores) in enumerate(lr_pairs.items()):
            if i >= d_niche:
                break
            niche_features[:, i] = scores[:N_epi]
        return niche_features
