"""
Visualization utilities for PRISM.

UMAP comparisons, ablation heatmaps, training curves, etc.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Dict, List
import os


def plot_umap_comparison(
    embeddings_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    label_names: Dict[int, str] = None,
    save_path: str = "figures/umap_comparison.png",
    figsize: tuple = None,
):
    """Plot UMAP comparison across methods.

    Shows side-by-side UMAP plots for PCA, Harmony, and PRISM
    embeddings, colored by fate label.

    Args:
        embeddings_dict: {method_name: (N, d) embeddings}
        labels: (N,) fate labels
        label_names: {label_int: label_name}
        save_path: path to save figure
    """
    from umap import UMAP

    if label_names is None:
        # Generate dynamically from unique labels
        unique_labels = sorted(np.unique(labels))
        label_names = {l: str(l) for l in unique_labels}

    n_methods = len(embeddings_dict)
    if figsize is None:
        figsize = (5 * n_methods, 4)

    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]

    # Generate colors dynamically for any number of labels
    _default_colors = ["#cccccc", "#999999", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c"]
    unique_labels = sorted(np.unique(labels))
    colors = {}
    for i, l in enumerate(unique_labels):
        colors[l] = _default_colors[i % len(_default_colors)]

    for ax, (method_name, emb) in zip(axes, embeddings_dict.items()):
        # Compute UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        umap_coords = reducer.fit_transform(emb)

        # Plot each label
        for label in sorted(np.unique(labels)):
            mask = labels == label
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=colors.get(label, "#000000"),
                label=label_names.get(label, str(label)),
                s=3,
                alpha=0.5,
                rasterized=True,
            )

        ax.set_title(method_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=7, markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP comparison to {save_path}")


def plot_ablation_heatmap(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: str = "figures/ablation_heatmap.png",
):
    """Plot ablation study results as a heatmap.

    Args:
        results: {ablation_name: {metric: value}}
        metrics: list of metrics to show
        save_path: path to save figure
    """
    if metrics is None:
        metrics = ["ARI", "AMI", "NMI", "ASW", "RF_F1_macro", "RF_AUROC"]

    methods = list(results.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    # Build matrix
    matrix = np.zeros((n_methods, n_metrics))
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            matrix[i, j] = results[method].get(metric, 0)

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.2), max(6, n_methods * 0.5)))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(methods, fontsize=9)

    # Add text annotations
    for i in range(n_methods):
        for j in range(n_metrics):
            text = f"{matrix[i, j]:.3f}"
            color = "white" if matrix[i, j] < 0.3 or matrix[i, j] > 0.8 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("PRISM Ablation Study", fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation heatmap to {save_path}")


def plot_training_curves(
    history: List[Dict],
    save_path: str = "figures/training_curves.png",
):
    """Plot training loss curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["loss"] for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    alphas = [h.get("alpha", 0) for h in history]
    temperatures = [h.get("temperature", 0.07) for h in history]
    mi_values = [h.get("mine_mi", 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curves
    axes[0, 0].plot(epochs, train_loss, label="Train", color="#2ecc71")
    axes[0, 0].plot(epochs, val_loss, label="Val", color="#e74c3c")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()

    # Curriculum alpha
    axes[0, 1].plot(epochs, alphas, color="#3498db")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("α")
    axes[0, 1].set_title("Hard-Negative Curriculum")

    # Temperature
    axes[1, 0].plot(epochs, temperatures, color="#9b59b6")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("τ")
    axes[1, 0].set_title("Learnable Temperature")

    # Mutual Information
    axes[1, 1].plot(epochs, mi_values, color="#e67e22")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MI estimate")
    axes[1, 1].set_title("MINE MI Estimate")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_discriminator_genes(
    gene_df,
    n_top: int = 20,
    save_path: str = "figures/discriminator_genes.png",
):
    """Plot top discriminator genes from horseshoe DE."""
    top = gene_df.head(n_top)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Effect size
    colors = ["#e74c3c" if b > 0 else "#3498db" for b in top["beta_fate_mean"]]
    axes[0].barh(range(n_top), top["beta_fate_mean"].values, color=colors)
    axes[0].set_yticks(range(n_top))
    axes[0].set_yticklabels(top["gene"].values, fontsize=8)
    axes[0].set_xlabel("Fate coefficient (β₁)")
    axes[0].set_title("Effect Size")
    axes[0].invert_yaxis()

    # PIP
    axes[1].barh(range(n_top), top["posterior_inclusion_prob"].values, color="#2ecc71")
    axes[1].set_yticks(range(n_top))
    axes[1].set_yticklabels(top["gene"].values, fontsize=8)
    axes[1].set_xlabel("Posterior Inclusion Probability")
    axes[1].set_title("PIP")
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()

    plt.suptitle("PRISM-Resolve: Top Cryptic Discriminator Genes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved discriminator genes plot to {save_path}")
