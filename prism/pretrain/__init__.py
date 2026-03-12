"""Perturbation-Contrastive Pre-training (PCP) for PRISM foundation model.

Pre-trains a transformer encoder on diverse perturbation datasets with:
- Perturbation-contrastive loss (cells with same perturbation cluster together)
- Masked gene prediction (BERT-style MLM adapted for gene expression)
"""

from .config import PCPConfig
from .model import PCPEncoder
from .dataset import CorpusDataset, PerturbationBatchSampler
from .trainer import PCPTrainer
