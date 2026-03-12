"""
Cross-system validation for PRISM.

Loads results from each biological system's output directory,
computes comparison metrics, and generates a cross-system summary.
"""

import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path


def load_system_results(
    systems: List[str] = None,
    base_dir: str = "data/processed",
    results_dir: str = "results",
) -> Dict[str, Dict]:
    """Load pipeline results for each system.

    Args:
        systems: List of system names (e.g., ["skin", "pancreas", "cortex", "hsc"])
        base_dir: Base directory for processed data
        results_dir: Directory containing per-system result files

    Returns:
        Dict mapping system name to result dict
    """
    if systems is None:
        systems = [
            "skin", "pancreas", "cortex", "hsc",
            "intestine", "cardiac", "neural_crest", "lung", "thcell", "oligo",
        ]

    system_results = {}

    for system in systems:
        result = {"system": system}

        # Try loading adata
        adata_path = os.path.join(base_dir, system, "adata_processed.h5ad")
        if not os.path.exists(adata_path):
            # Fall back to default location
            adata_path = os.path.join(base_dir, "adata_processed.h5ad")

        if os.path.exists(adata_path):
            try:
                adata = ad.read_h5ad(adata_path)
                result["n_cells"] = adata.shape[0]
                result["n_genes"] = adata.shape[1]
                result["has_prism_embedding"] = "X_prism" in adata.obsm

                if "fate_label" in adata.obs:
                    result["fate_distribution"] = adata.obs["fate_label"].value_counts().to_dict()
                if "fate_int" in adata.obs:
                    result["n_fate_categories"] = len(adata.obs["fate_int"].unique())
            except Exception as e:
                result["load_error"] = str(e)

        # Try loading metrics from JSON
        metrics_path = os.path.join(results_dir, system, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                result["metrics"] = json.load(f)

        # Try loading DE results
        de_path = os.path.join(results_dir, system, "de_results.csv")
        if os.path.exists(de_path):
            de_df = pd.read_csv(de_path)
            result["n_discriminators_pip05"] = int((de_df["posterior_inclusion_prob"] > 0.5).sum())
            result["n_discriminators_pip09"] = int((de_df["posterior_inclusion_prob"] > 0.9).sum())
            result["top_genes"] = de_df.head(10)["gene"].tolist()

        system_results[system] = result

    return system_results


def load_baseline_results(
    systems: List[str] = None,
    base_dir: str = "data/processed",
) -> Dict[str, Dict[str, Dict]]:
    """Load baseline comparison results for each system.

    Looks for baseline_results.json in each system's processed directory.

    Args:
        systems: List of system names
        base_dir: Base directory for processed data

    Returns:
        Dict mapping system name to dict of {method_name: metrics_dict}
    """
    if systems is None:
        systems = [
            "skin", "pancreas", "cortex", "hsc",
            "intestine", "cardiac", "neural_crest", "lung", "thcell", "oligo",
        ]

    all_baselines = {}
    for system in systems:
        baselines_path = os.path.join(base_dir, system, "baseline_results.json")
        if not os.path.exists(baselines_path):
            # Try default location for skin
            baselines_path = os.path.join(base_dir, "baseline_results.json")

        if os.path.exists(baselines_path):
            try:
                with open(baselines_path, "r") as f:
                    all_baselines[system] = json.load(f)
            except Exception:
                pass

        # Also load foundation model results if available
        foundation_path = os.path.join(base_dir, system, "foundation_results.json")
        if not os.path.exists(foundation_path):
            foundation_path = os.path.join(base_dir, "foundation_results.json")

        if os.path.exists(foundation_path):
            try:
                with open(foundation_path, "r") as f:
                    foundation = json.load(f)
                if system not in all_baselines:
                    all_baselines[system] = {}
                for method_name, method_metrics in foundation.items():
                    if isinstance(method_metrics, dict) and "error" not in method_metrics:
                        all_baselines[system][method_name] = method_metrics
            except Exception:
                pass

    return all_baselines


def compute_comparison_table(
    system_results: Dict[str, Dict],
    baseline_results: Optional[Dict[str, Dict[str, Dict]]] = None,
) -> pd.DataFrame:
    """Build cross-system comparison table.

    Includes PRISM metrics and optionally baseline method metrics.

    Returns:
        DataFrame with one row per (system, method), columns for key metrics.
    """
    rows = []
    for system, result in system_results.items():
        # PRISM row
        row = {
            "System": system,
            "Method": "PRISM",
            "N_cells": result.get("n_cells", 0),
            "N_genes": result.get("n_genes", 0),
            "N_fates": result.get("n_fate_categories", 0),
        }

        metrics = result.get("metrics", {})
        row["ARI"] = metrics.get("ARI", np.nan)
        row["RF_AUROC"] = metrics.get("RF_AUROC", np.nan)
        row["RF_F1_macro"] = metrics.get("RF_F1_macro", np.nan)
        row["ASW"] = metrics.get("ASW", np.nan)
        row["N_discriminators"] = result.get("n_discriminators_pip05", 0)

        rows.append(row)

        # Baseline rows for this system
        if baseline_results and system in baseline_results:
            sys_baselines = baseline_results[system]
            for method_name, method_metrics in sys_baselines.items():
                if not isinstance(method_metrics, dict) or "error" in method_metrics:
                    continue
                if method_metrics.get("simulated", False):
                    continue  # Skip simulated baselines

                brow = {
                    "System": system,
                    "Method": method_name,
                    "N_cells": result.get("n_cells", 0),
                    "N_genes": result.get("n_genes", 0),
                    "N_fates": result.get("n_fate_categories", 0),
                    "ARI": method_metrics.get("ARI", np.nan),
                    "RF_AUROC": method_metrics.get("RF_AUROC", np.nan),
                    "RF_F1_macro": method_metrics.get("RF_F1_macro", np.nan),
                    "ASW": method_metrics.get("ASW", np.nan),
                    "N_discriminators": 0,
                }
                rows.append(brow)

    df = pd.DataFrame(rows)
    return df


def run_cross_system_analysis(
    adata_dict: Optional[Dict[str, ad.AnnData]] = None,
    systems: List[str] = None,
    save_dir: str = "figures",
) -> pd.DataFrame:
    """Run cross-system analysis from loaded AnnData objects.

    If adata_dict is provided, computes metrics directly.
    Otherwise loads from disk via load_system_results.

    Args:
        adata_dict: Optional dict mapping system names to AnnData objects
        systems: System names to analyze
        save_dir: Directory for saving figures

    Returns:
        Comparison table as DataFrame
    """
    from ..utils.metrics import compute_clustering_metrics, compute_classification_metrics

    if systems is None:
        systems = [
            "skin", "pancreas", "cortex", "hsc",
            "intestine", "cardiac", "neural_crest", "lung", "thcell", "oligo",
        ]

    if adata_dict is not None:
        system_results = {}
        for system in systems:
            if system not in adata_dict:
                continue

            adata = adata_dict[system]
            result = {
                "system": system,
                "n_cells": adata.shape[0],
                "n_genes": adata.shape[1],
            }

            if "fate_int" in adata.obs:
                labels = adata.obs["fate_int"].values
                result["n_fate_categories"] = len(np.unique(labels))

                # Compute metrics if PRISM embeddings exist
                if "X_prism" in adata.obsm:
                    embeddings = adata.obsm["X_prism"]

                    clustering = compute_clustering_metrics(embeddings, labels)
                    classification = compute_classification_metrics(embeddings, labels)

                    result["metrics"] = {**clustering, **classification}

            system_results[system] = result
    else:
        system_results = load_system_results(systems)

    # Load baseline results if available
    baseline_results = load_baseline_results(systems)

    # Build comparison table
    comparison = compute_comparison_table(system_results, baseline_results)
    print("\n=== Cross-System Comparison ===")
    print(comparison.to_string(index=False))

    # Save table
    os.makedirs(save_dir, exist_ok=True)
    comparison.to_csv(os.path.join(save_dir, "cross_system_comparison.csv"), index=False)

    # Generate summary figure
    plot_cross_system_summary(comparison, save_dir=save_dir)

    return comparison


def plot_cross_system_summary(
    comparison: pd.DataFrame,
    save_dir: str = "figures",
):
    """Generate cross-system summary figure.

    Creates a grouped bar chart comparing key metrics across systems.
    """
    metrics_to_plot = ["ARI", "RF_AUROC", "ASW"]
    available_metrics = [m for m in metrics_to_plot if m in comparison.columns and comparison[m].notna().any()]

    if not available_metrics:
        print("No metrics available for cross-system plot")
        return

    systems = comparison["System"].values
    n_systems = len(systems)
    n_metrics = len(available_metrics)

    fig, ax = plt.subplots(figsize=(max(8, n_systems * 2), 5))

    x = np.arange(n_systems)
    width = 0.8 / n_metrics

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for i, metric in enumerate(available_metrics):
        values = comparison[metric].fillna(0).values
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i % len(colors)])

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel("Biological System", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("PRISM Cross-System Validation", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cross_system_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cross-system comparison to {save_path}")
