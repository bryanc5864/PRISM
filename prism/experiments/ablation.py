"""
Ablation study runner for PRISM.

Tests 10 ablation variants to quantify contribution of each component:
1. PRISM-full (reference)
2. w/o hard-negatives (α=0)
3. w/o curriculum (α=α_max from epoch 0)
4. w/o info regularizer (λ=0)
5. w/o LoRA (full fine-tuning)
6. w/o pretrained init (random)
7. w/o reconstruction loss (μ=0)
8. w/o niche context
9. w/o condition embedding
10. w/ horseshoe → BH (frequentist DE)
"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from ..models.encoder import PRISMEncoder
from ..training.trainer import PRISMTrainer
from ..utils.metrics import compute_all_metrics


def run_ablation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_labels: np.ndarray,
    base_config: dict,
    device: str = "cuda:0",
    n_epochs: int = 30,
) -> Dict[str, Dict]:
    """Run all ablation experiments.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        test_labels: Ground truth labels for test set
        base_config: Base configuration dict
        device: CUDA device
        n_epochs: Training epochs per ablation

    Returns:
        Dict mapping ablation name to metrics dict
    """
    results = {}

    ablation_configs = _get_ablation_configs(base_config)

    for name, config in ablation_configs.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")

        try:
            encoder = PRISMEncoder(
                n_genes=config.get("n_genes", 2000),
                n_bins=config.get("n_expression_bins", 51),
                d_model=config.get("d_model", 512),
                n_layers=config.get("n_layers", 12),
                n_heads=config.get("n_heads", 8),
                d_ff=config.get("d_ff", 1024),
                d_output=config.get("d_output", 256),
                dropout=config.get("dropout", 0.1),
                lora_rank=config.get("lora_rank", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.1),
                projection_dims=config.get("projection_dims", [512, 256, 128]),
            )

            trainer = PRISMTrainer(encoder, config, device=device)
            train_result = trainer.train(
                train_loader, val_loader,
                n_epochs=n_epochs,
                patience=config.get("early_stopping_patience", 10),
                checkpoint_dir=f"checkpoints/ablation_{name}",
            )

            # Extract embeddings and compute metrics
            embeddings, labels, genotypes = trainer.extract_embeddings(test_loader)
            metrics = compute_all_metrics(embeddings, test_labels, method_name=name)
            metrics["training_time"] = train_result.get("total_time_seconds", 0)
            metrics["n_epochs_trained"] = train_result.get("n_epochs_trained", 0)
            metrics["best_val_loss"] = train_result.get("best_val_loss", float("inf"))

            results[name] = metrics
            print(f"  ARI={metrics.get('ARI', 0):.4f}, AMI={metrics.get('AMI', 0):.4f}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"method": name, "error": str(e)}

    return results


def _get_ablation_configs(base_config: dict) -> Dict[str, dict]:
    """Generate ablation-specific configurations."""

    ablations = {}

    # 1. Full model (reference)
    ablations["PRISM-full"] = copy.deepcopy(base_config)

    # 2. Without hard-negatives (α=0)
    config = copy.deepcopy(base_config)
    config["alpha_max"] = 0.0
    ablations["w/o hard-neg"] = config

    # 3. Without curriculum (α=α_max from start)
    config = copy.deepcopy(base_config)
    config["curriculum_warmup_epochs"] = 0
    ablations["w/o curriculum"] = config

    # 4. Without info regularizer
    config = copy.deepcopy(base_config)
    config["info_reg_lambda"] = 0.0
    ablations["w/o info-reg"] = config

    # 5. Without LoRA (full fine-tuning)
    config = copy.deepcopy(base_config)
    config["lora_rank"] = 0  # Signal to use full params
    config["full_finetune"] = True
    ablations["w/o LoRA"] = config

    # 6. Without pretrained init (random)
    config = copy.deepcopy(base_config)
    config["random_init"] = True
    ablations["w/o pretrain"] = config

    # 7. Without reconstruction loss
    config = copy.deepcopy(base_config)
    config["recon_weight"] = 0.0
    ablations["w/o recon"] = config

    # 8. Without niche context
    config = copy.deepcopy(base_config)
    config["use_niche"] = False
    ablations["w/o niche"] = config

    # 9. Without condition embedding
    config = copy.deepcopy(base_config)
    config["no_condition"] = True
    ablations["w/o condition"] = config

    # 10. Horseshoe → BH (different DE method)
    config = copy.deepcopy(base_config)
    config["de_method"] = "bh"
    ablations["BH instead"] = config

    return ablations
