"""
MINE (Mutual Information Neural Estimation) for PRISM.

Used as the information-preserving regularizer:
R_info = -λ · Î_MINE(f_θ(X); Y)

Prevents the encoder from discarding low-variance but informative
dimensions by directly estimating and maximizing mutual information
between the learned representation and fate labels.

Reference: Belghazi et al., "Mutual Information Neural Estimation", ICML 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MINEStatisticsNetwork(nn.Module):
    """Statistics network T(x, y) for MINE.

    A small MLP that takes concatenated (embedding, label) pairs
    and outputs a scalar score. The MINE bound on MI is:
    I(X; Y) ≥ E[T(x, y)] - log(E[exp(T(x', y))])
    where x' is drawn from the marginal p(x).
    """

    def __init__(self, input_dim: int, label_dim: int, hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        layers = []
        in_d = input_dim + label_dim
        for h_d in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.ReLU(),
            ])
            in_d = h_d
        layers.append(nn.Linear(in_d, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d) embeddings
            y: (B, label_dim) one-hot or continuous labels

        Returns:
            scores: (B,) scalar scores
        """
        xy = torch.cat([x, y], dim=-1)
        return self.network(xy).squeeze(-1)


class MINEEstimator(nn.Module):
    """Mutual Information Neural Estimator.

    Estimates I(f_θ(X); Y) using the Donsker-Varadhan representation:
    I(X; Y) = sup_T { E_joint[T(x,y)] - log(E_marginal[exp(T(x,y))]) }

    Uses exponential moving average for the log-sum-exp term
    to reduce bias (as in Belghazi et al., 2018).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        n_labels: int = 4,
        hidden_dims: list = None,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.n_labels = n_labels
        self.ema_decay = ema_decay

        self.stats_net = MINEStatisticsNetwork(
            input_dim=embedding_dim,
            label_dim=n_labels,
            hidden_dims=hidden_dims or [128, 64],
        )

        # EMA for bias correction
        self.register_buffer("ema", torch.tensor(1.0))

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, d)
        labels: torch.Tensor,      # (B,) integer labels
    ) -> Tuple[torch.Tensor, dict]:
        """Estimate mutual information I(embeddings; labels).

        Args:
            embeddings: (B, d) learned representations
            labels: (B,) integer fate labels

        Returns:
            mi_estimate: scalar MI lower bound
            metrics: dict with MI value and diagnostic info
        """
        B = embeddings.shape[0]
        device = embeddings.device

        # One-hot encode labels
        y_onehot = F.one_hot(labels, self.n_labels).float()

        # Joint distribution: (x_i, y_i) pairs
        t_joint = self.stats_net(embeddings, y_onehot)

        # Marginal distribution: shuffle labels to break dependence
        perm = torch.randperm(B, device=device)
        y_marginal = y_onehot[perm]
        t_marginal = self.stats_net(embeddings, y_marginal)

        # MINE lower bound with EMA bias correction
        joint_mean = t_joint.mean()
        exp_marginal = torch.exp(t_marginal)

        if self.training:
            # EMA for stable log-mean-exp estimation
            self.ema = self.ema_decay * self.ema + (1 - self.ema_decay) * exp_marginal.mean().detach()
            log_mean_exp = torch.log(exp_marginal.mean() + 1e-8) - \
                           torch.log(self.ema + 1e-8) + \
                           (exp_marginal.mean().detach() / (self.ema + 1e-8)).log()
        else:
            log_mean_exp = torch.log(exp_marginal.mean() + 1e-8)

        mi_estimate = joint_mean - log_mean_exp

        metrics = {
            "mi_estimate": mi_estimate.item(),
            "joint_mean": joint_mean.item(),
            "marginal_log_mean_exp": log_mean_exp.item(),
        }

        return mi_estimate, metrics

    def compute_regularizer(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        lambda_info: float = 0.1,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute information-preserving regularizer.

        R_info = -λ · Î_MINE(f_θ(X); Y)

        Negative because we want to MAXIMIZE MI (minimize negative MI).
        """
        mi_estimate, metrics = self.forward(embeddings, labels)
        regularizer = -lambda_info * mi_estimate
        metrics["regularizer"] = regularizer.item()
        return regularizer, metrics
