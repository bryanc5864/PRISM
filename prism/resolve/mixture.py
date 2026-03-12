"""
Bayesian Gaussian Mixture Model for fate assignment in PRISM-Resolve.

Assigns probabilistic fate memberships to cells in the learned
embedding space: P(fate = k | z_i) for k ∈ {eccrine, hair, uncommitted}.

Uses semi-supervised anchors from clearly committed cells:
- Late Epi3 in WT → eccrine
- Late Epi3 in En1-cKO → hair
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from typing import Optional, Tuple, Dict


class BayesianFateMixture:
    """Bayesian Gaussian Mixture for fate probability assignment.

    In the PRISM embedding space, cells cluster by fate rather than
    by shared transcriptional programs. This mixture model assigns
    calibrated probabilities P(eccrine | z), P(hair | z), P(uncommitted | z).

    Semi-supervised: known labels constrain the mixture components.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        max_iter: int = 200,
        n_init: int = 10,
        random_state: int = 42,
        fate_names: Optional[list] = None,
    ):
        self.n_components = n_components
        self.component_names = fate_names or ["uncommitted", "eccrine", "hair"]

        self.model = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0 / n_components,
        )

        self.is_fitted = False
        self.component_to_fate = {}

    def fit(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        label_mask: Optional[np.ndarray] = None,
    ):
        """Fit the mixture model with optional semi-supervised labels.

        Args:
            embeddings: (N, d) cell embeddings from PRISM-Encode
            labels: (N,) fate labels (any integer encoding; known fates indicated by label_mask)
            label_mask: (N,) boolean mask of cells with known labels
        """
        # Fit GMM on all data
        self.model.fit(embeddings)

        # If labels provided, match components to fates
        if labels is not None and label_mask is not None:
            self._match_components_to_fates(embeddings, labels, label_mask)

        self.is_fitted = True

    def _match_components_to_fates(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_mask: np.ndarray,
    ):
        """Match GMM components to fate labels using labeled cells.

        Maps components to output indices: 0=uncommitted, 1=eccrine, 2=hair
        regardless of input label encoding.
        """
        # Get component assignments for labeled cells
        labeled_embeddings = embeddings[label_mask]
        labeled_labels = labels[label_mask]
        component_probs = self.model.predict_proba(labeled_embeddings)

        # Map input labels to output fate indices (0=uncommitted, 1..N=known fates)
        unique_labels = sorted(np.unique(labeled_labels[labeled_labels >= 0]))

        # Build label-to-fate-index mapping: map N unique labels to fate indices 1..N
        label_to_fate_idx = {}
        for i, label in enumerate(unique_labels):
            label_to_fate_idx[label] = i + 1  # fate index 1, 2, ..., N

        self.component_to_fate = {}
        used_components = set()

        for label in unique_labels:
            mask = labeled_labels == label
            avg_probs = component_probs[mask].mean(axis=0)

            # Exclude already-assigned components
            for used in used_components:
                avg_probs[used] = -1

            best_component = avg_probs.argmax()
            fate_idx = label_to_fate_idx.get(label, 0)
            self.component_to_fate[best_component] = fate_idx
            used_components.add(best_component)

        # Assign remaining components to "uncommitted" (index 0)
        for c in range(self.n_components):
            if c not in self.component_to_fate:
                self.component_to_fate[c] = 0

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict fate probabilities for cells.

        Returns:
            probs: (N, n_fates) array with columns [uncommitted, eccrine, hair]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction")

        component_probs = self.model.predict_proba(embeddings)

        # Remap component probabilities to fate probabilities
        n_fates = max(self.component_to_fate.values()) + 1 if self.component_to_fate else self.n_components
        fate_probs = np.zeros((embeddings.shape[0], n_fates))

        if self.component_to_fate:
            for comp, fate in self.component_to_fate.items():
                if comp < component_probs.shape[1]:
                    fate_probs[:, fate] += component_probs[:, comp]
        else:
            fate_probs[:, :self.n_components] = component_probs

        # Normalize
        fate_probs = fate_probs / (fate_probs.sum(axis=1, keepdims=True) + 1e-8)

        return fate_probs

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict most likely fate for each cell."""
        probs = self.predict_proba(embeddings)
        return probs.argmax(axis=1)

    def get_fate_scores(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """Get named fate probability scores."""
        probs = self.predict_proba(embeddings)
        result = {}
        for i, name in enumerate(self.component_names):
            result[name] = probs[:, i] if probs.shape[1] > i else np.zeros(len(probs))
        return result

    def compute_entropy(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute fate decision entropy for each cell.

        High entropy = undecided progenitor.
        Low entropy = committed to a specific fate.
        """
        probs = self.predict_proba(embeddings)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return entropy
