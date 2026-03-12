"""
Hard-negative curriculum for PRISM training.

Implements the curriculum schedule:
α(t) = α_max · min(1, t / T₀)

Early training captures broad cell-type distinctions,
later training refines the boundary between cryptically similar populations.
"""

import numpy as np
from typing import Dict


class HardNegativeCurriculum:
    """Manage hard-negative curriculum schedule during training.

    The curriculum gradually increases the difficulty of negative examples:
    1. Epoch 0: α=0 (uniform negatives, learn broad distinctions)
    2. Epochs 1-T₀: α linearly increases (progressively harder negatives)
    3. Epoch T₀+: α=α_max (full hard-negative weighting)
    """

    def __init__(
        self,
        alpha_max: float = 2.0,
        warmup_epochs: int = 10,
        schedule: str = "linear",
    ):
        self.alpha_max = alpha_max
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule
        self.history = []

    def get_alpha(self, epoch: int) -> float:
        """Get current hard-negative weight α for given epoch."""
        if self.schedule == "linear":
            alpha = self.alpha_max * min(1.0, epoch / max(self.warmup_epochs, 1))
        elif self.schedule == "cosine":
            if epoch >= self.warmup_epochs:
                alpha = self.alpha_max
            else:
                alpha = self.alpha_max * (1 - np.cos(np.pi * epoch / self.warmup_epochs)) / 2
        elif self.schedule == "step":
            alpha = self.alpha_max if epoch >= self.warmup_epochs else 0.0
        else:
            alpha = self.alpha_max * min(1.0, epoch / max(self.warmup_epochs, 1))

        self.history.append(alpha)
        return alpha

    def get_schedule_summary(self) -> Dict:
        """Return summary of the curriculum schedule."""
        return {
            "alpha_max": self.alpha_max,
            "warmup_epochs": self.warmup_epochs,
            "schedule": self.schedule,
            "history": self.history,
        }
