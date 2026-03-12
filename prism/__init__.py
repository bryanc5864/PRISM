"""
PRISM: Progenitor Resolution via Invariance-Sensitive Modeling

A deep learning framework for resolving transcriptionally cryptic
cell fate decisions from perturbation scRNA-seq data.

Usage:
    import prism

    model = prism.PRISM(adata, condition_key="genotype")
    model.preprocess()
    model.fit(n_epochs=50)
    model.resolve(method="fast")
    discriminators = model.get_discriminators(pip_threshold=0.5)
"""

__version__ = "1.0.0"

from .api import PRISM
from .config import SystemConfig, SKIN_CONFIG

__all__ = ["PRISM", "SystemConfig", "SKIN_CONFIG"]
