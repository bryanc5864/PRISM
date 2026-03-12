"""
Sensitivity analysis for PRISM.

Tests robustness to:
1. Hyperparameter choices (α, λ, LoRA rank r, temperature)
2. Downsampling (25%, 50%, 75%)
3. Gene dropout (10%, 20%, 30%)
4. Label noise (5%, 10%, 20%)
"""

import numpy as np
import copy
from torch.utils.data import DataLoader, Subset
from typing import Dict, List

from ..models.encoder import PRISMEncoder
from ..training.trainer import PRISMTrainer
from ..utils.metrics import compute_all_metrics


def run_sensitivity(
    train_dataset,
    val_dataset,
    test_dataset,
    test_labels: np.ndarray,
    base_config: dict,
    device: str = "cuda:0",
    n_epochs: int = 20,
) -> Dict[str, Dict]:
    """Run all sensitivity analyses.

    Returns:
        Dict with results for each sensitivity experiment
    """
    results = {}

    # 1. Hyperparameter sweeps
    print("\n=== Hyperparameter Sensitivity ===")
    results["hyperparams"] = _hyperparameter_sweep(
        train_dataset, val_dataset, test_dataset,
        test_labels, base_config, device, n_epochs
    )

    # 2. Downsampling
    print("\n=== Downsampling Sensitivity ===")
    results["downsampling"] = _downsampling_analysis(
        train_dataset, val_dataset, test_dataset,
        test_labels, base_config, device, n_epochs
    )

    # 3. Gene dropout
    print("\n=== Gene Dropout Sensitivity ===")
    results["gene_dropout"] = _gene_dropout_analysis(
        train_dataset, val_dataset, test_dataset,
        test_labels, base_config, device, n_epochs
    )

    # 4. Label noise
    print("\n=== Label Noise Sensitivity ===")
    results["label_noise"] = _label_noise_analysis(
        train_dataset, val_dataset, test_dataset,
        test_labels, base_config, device, n_epochs
    )

    return results


def _hyperparameter_sweep(
    train_dataset, val_dataset, test_dataset,
    test_labels, base_config, device, n_epochs
) -> Dict:
    """Grid search over key hyperparameters."""
    sweep_params = {
        "alpha_max": [0.5, 1.0, 2.0, 4.0],
        "info_reg_lambda": [0.01, 0.1, 0.5, 1.0],
        "lora_rank": [4, 8, 16],
        "temperature_init": [0.05, 0.07, 0.1],
    }

    results = {}
    for param_name, values in sweep_params.items():
        param_results = {}
        for value in values:
            config = copy.deepcopy(base_config)
            config[param_name] = value

            key = f"{param_name}={value}"
            print(f"  Testing {key}...")

            try:
                metrics = _train_and_evaluate(
                    train_dataset, val_dataset, test_dataset,
                    test_labels, config, device, n_epochs
                )
                param_results[key] = metrics
            except Exception as e:
                param_results[key] = {"error": str(e)}

        results[param_name] = param_results

    return results


def _downsampling_analysis(
    train_dataset, val_dataset, test_dataset,
    test_labels, base_config, device, n_epochs
) -> Dict:
    """Test robustness to reduced sample sizes."""
    fractions = [0.25, 0.50, 0.75, 1.0]
    results = {}

    for frac in fractions:
        key = f"{int(frac*100)}%"
        print(f"  Testing downsampling to {key}...")

        n_samples = int(len(train_dataset) * frac)
        indices = np.random.choice(len(train_dataset), size=n_samples, replace=False)

        subset = Subset(train_dataset, indices)
        subset_loader = DataLoader(subset, batch_size=base_config.get("batch_size", 256),
                                   shuffle=True, num_workers=0)

        val_loader = DataLoader(val_dataset, batch_size=base_config.get("batch_size", 256),
                                shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=base_config.get("batch_size", 256),
                                 shuffle=False, num_workers=0)

        try:
            encoder = _build_encoder(base_config)
            trainer = PRISMTrainer(encoder, base_config, device=device)
            trainer.train(subset_loader, val_loader, n_epochs=n_epochs, patience=10,
                         checkpoint_dir=f"checkpoints/downsample_{key}")
            embeddings, _, _ = trainer.extract_embeddings(test_loader)
            metrics = compute_all_metrics(embeddings, test_labels, method_name=key)
            results[key] = metrics
        except Exception as e:
            results[key] = {"error": str(e)}

    return results


def _gene_dropout_analysis(
    train_dataset, val_dataset, test_dataset,
    test_labels, base_config, device, n_epochs
) -> Dict:
    """Test robustness to gene expression dropout."""
    dropout_rates = [0.0, 0.10, 0.20, 0.30]
    results = {}

    for rate in dropout_rates:
        key = f"dropout={int(rate*100)}%"
        print(f"  Testing {key}...")

        # Apply additional dropout to expression values
        config = copy.deepcopy(base_config)
        config["additional_dropout"] = rate

        try:
            metrics = _train_and_evaluate(
                train_dataset, val_dataset, test_dataset,
                test_labels, config, device, n_epochs
            )
            results[key] = metrics
        except Exception as e:
            results[key] = {"error": str(e)}

    return results


def _label_noise_analysis(
    train_dataset, val_dataset, test_dataset,
    test_labels, base_config, device, n_epochs
) -> Dict:
    """Test robustness to label noise."""
    noise_rates = [0.0, 0.05, 0.10, 0.20]
    results = {}

    for rate in noise_rates:
        key = f"noise={int(rate*100)}%"
        print(f"  Testing {key}...")

        config = copy.deepcopy(base_config)
        config["label_noise_rate"] = rate

        try:
            metrics = _train_and_evaluate(
                train_dataset, val_dataset, test_dataset,
                test_labels, config, device, n_epochs
            )
            results[key] = metrics
        except Exception as e:
            results[key] = {"error": str(e)}

    return results


def _build_encoder(config: dict) -> PRISMEncoder:
    """Build encoder from config."""
    return PRISMEncoder(
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


def _train_and_evaluate(
    train_dataset, val_dataset, test_dataset,
    test_labels, config, device, n_epochs
) -> Dict:
    """Helper to train and evaluate a configuration."""
    batch_size = config.get("batch_size", 256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    encoder = _build_encoder(config)
    trainer = PRISMTrainer(encoder, config, device=device)
    trainer.train(train_loader, val_loader, n_epochs=n_epochs, patience=10,
                 checkpoint_dir="checkpoints/sensitivity_temp")

    embeddings, _, _ = trainer.extract_embeddings(test_loader)
    return compute_all_metrics(embeddings, test_labels)
