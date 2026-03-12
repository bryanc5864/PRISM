"""
Hard-Negative Weighted InfoNCE Loss for PRISM.

Key innovation: importance-weighted negative sampling that amplifies
gradients from transcriptionally similar but fate-divergent cell pairs.

Fully vectorized — no Python loops over anchors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class HardNegativeInfoNCE(nn.Module):
    """Hard-negative weighted InfoNCE contrastive loss (vectorized).

    L_PRISM = -(1/N) Σᵢ log[ exp(sim(zᵢ, zᵢ⁺)/τ) / Σⱼ wᵢⱼ · exp(sim(zᵢ, zⱼ)/τ) ]

    where wᵢⱼ = exp(α · sim₀(xᵢ, xⱼ)) for hard-negative weighting.
    """

    def __init__(
        self,
        temperature_init: float = 0.07,
        alpha: float = 0.0,
        known_fate_threshold: int = 2,
    ):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(temperature_init)))
        self.alpha = alpha
        self.known_fate_threshold = known_fate_threshold

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(
        self,
        z: torch.Tensor,           # (B, d) L2-normalized embeddings
        fate_labels: torch.Tensor,  # (B,) fate labels
        raw_sim: Optional[torch.Tensor] = None,  # (B, B) raw expression similarity
        genotypes: Optional[torch.Tensor] = None, # (B,) genotype labels
    ) -> Tuple[torch.Tensor, dict]:
        B = z.shape[0]
        device = z.device

        # Similarity matrix in embedding space
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (B, B)

        # --- Build positive mask (vectorized) ---
        known_mask = (fate_labels >= self.known_fate_threshold)
        fate_match = fate_labels.unsqueeze(0) == fate_labels.unsqueeze(1)
        eye = torch.eye(B, device=device, dtype=torch.bool)

        positive_mask = fate_match & known_mask.unsqueeze(0) & known_mask.unsqueeze(1) & ~eye

        # For unknown-fate cells, pair within same genotype
        if genotypes is not None:
            unknown = ~known_mask
            geno_match = genotypes.unsqueeze(0) == genotypes.unsqueeze(1)
            unknown_pos = unknown.unsqueeze(0) & unknown.unsqueeze(1) & geno_match & ~eye
            positive_mask = positive_mask | unknown_pos

        # --- Build negative mask ---
        negative_mask = ~positive_mask & ~eye

        # --- Hard-negative weights ---
        if raw_sim is not None and self.alpha > 0:
            neg_weights = torch.exp(self.alpha * raw_sim)
        else:
            neg_weights = torch.ones(B, B, device=device)

        # --- Vectorized loss computation ---
        # For each anchor i, we need:
        #   pos_score = mean of sim_matrix[i, positive_j]
        #   neg_denom = sum of neg_weights[i,j] * exp(sim_matrix[i,j]) for negative j

        # Positive: masked mean similarity
        pos_mask_f = positive_mask.float()
        pos_count = pos_mask_f.sum(dim=1).clamp(min=1)           # (B,)
        pos_sim_sum = (sim_matrix * pos_mask_f).sum(dim=1)        # (B,)
        pos_sim_mean = pos_sim_sum / pos_count                    # (B,)

        # Negative: weighted log-sum-exp
        # Set masked-out positions to -inf so exp() → 0
        neg_logits = sim_matrix + torch.log(neg_weights + 1e-10)  # (B, B)
        neg_logits = neg_logits.masked_fill(~negative_mask, float('-inf'))
        neg_logsumexp = torch.logsumexp(neg_logits, dim=1)       # (B,)

        # InfoNCE per anchor: -pos + log(exp(pos) + sum_neg)
        # = -pos + log(exp(pos) + exp(neg_logsumexp))
        # = -pos + logsumexp(pos, neg_logsumexp)
        combined = torch.stack([pos_sim_mean, neg_logsumexp], dim=1)  # (B, 2)
        log_denom = torch.logsumexp(combined, dim=1)                  # (B,)
        per_anchor_loss = -pos_sim_mean + log_denom                   # (B,)

        # Only include anchors that have at least one positive
        has_positive = pos_mask_f.sum(dim=1) > 0
        if has_positive.any():
            loss = per_anchor_loss[has_positive].mean()
        else:
            return self._fallback_loss(sim_matrix, eye)

        with torch.no_grad():
            metrics = {
                "temperature": self.temperature.item(),
                "n_valid_anchors": int(has_positive.sum().item()),
                "loss": loss.item(),
            }

        return loss, metrics

    def _fallback_loss(self, sim_matrix, eye):
        """Standard InfoNCE when no valid positive pairs exist."""
        B = sim_matrix.shape[0]
        labels = torch.arange(B, device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix.masked_fill(eye, float('-inf')), labels)
        return loss, {"temperature": self.temperature.item(), "fallback": True}


class CurriculumScheduler:
    """Curriculum schedule for hard-negative weight α."""

    def __init__(self, alpha_max: float = 2.0, warmup_epochs: int = 10):
        self.alpha_max = alpha_max
        self.warmup_epochs = warmup_epochs

    def get_alpha(self, epoch: int) -> float:
        return self.alpha_max * min(1.0, epoch / max(self.warmup_epochs, 1))

    def update_loss(self, loss_fn: HardNegativeInfoNCE, epoch: int):
        loss_fn.alpha = self.get_alpha(epoch)


def compute_raw_similarity_matrix(
    raw_expression: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity matrix from raw expression for hard-negative weighting."""
    norm = F.normalize(raw_expression, dim=-1)
    return torch.matmul(norm, norm.T)
