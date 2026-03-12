#!/usr/bin/env python
"""Generate UMAP visualizations comparing PRISM vs PCA vs scGPT vs Geneformer.

Produces per-system comparison figures and a combined 4x4 grid figure.
"""

import os
import sys
import warnings
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from umap import UMAP

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYSTEMS = ["skin", "pancreas", "cortex", "hsc"]
METHODS = ["PRISM", "PCA", "scGPT", "Geneformer"]

SYSTEM_DATA = {
    "skin": {
        "adata": "data/processed/adata_processed.h5ad",
        "scgpt": "data/processed/scgpt_embeddings.npy",
        "geneformer": "data/processed/geneformer_embeddings.npy",
    },
    "pancreas": {
        "adata": "data/processed/pancreas/adata_processed.h5ad",
        "scgpt": "data/processed/pancreas/scgpt_embeddings.npy",
        "geneformer": "data/processed/pancreas/geneformer_embeddings.npy",
    },
    "cortex": {
        "adata": "data/processed/cortex/adata_processed.h5ad",
        "scgpt": "data/processed/cortex/scgpt_embeddings.npy",
        "geneformer": "data/processed/cortex/geneformer_embeddings.npy",
    },
    "hsc": {
        "adata": "data/processed/hsc/adata_processed.h5ad",
        "scgpt": "data/processed/hsc/scgpt_embeddings.npy",
        "geneformer": "data/processed/hsc/geneformer_embeddings.npy",
    },
}

# Display names for systems
SYSTEM_DISPLAY = {
    "skin": "Skin",
    "pancreas": "Pancreas",
    "cortex": "Cortex",
    "hsc": "HSC",
}

# Per-system fate label display names and consistent ordering
FATE_DISPLAY = {
    "skin": {
        "eccrine": "Eccrine",
        "hair": "Hair follicle",
        "non_appendage": "Non-appendage",
        "undetermined": "Undetermined",
    },
    "pancreas": {
        "alpha": "Alpha",
        "beta": "Beta",
        "delta": "Delta",
        "non_endocrine": "Non-endocrine",
        "undetermined": "Undetermined",
    },
    "cortex": {
        "upper_layer": "Upper layer",
        "deep_layer": "Deep layer",
        "non_neuronal": "Non-neuronal",
        "undetermined": "Undetermined",
    },
    "hsc": {
        "erythroid": "Erythroid",
        "myeloid": "Myeloid",
        "lymphoid": "Lymphoid",
        "undetermined": "Undetermined",
    },
}

# Color palettes per system -- fate-specific colors, undetermined always grey
FATE_COLORS = {
    "skin": {
        "eccrine": "#2196F3",      # blue
        "hair": "#E91E63",         # pink/magenta
        "non_appendage": "#9E9E9E",# grey
        "undetermined": "#E0E0E0", # light grey
    },
    "pancreas": {
        "alpha": "#4CAF50",        # green
        "beta": "#FF9800",         # orange
        "delta": "#9C27B0",        # purple
        "non_endocrine": "#9E9E9E",# grey
        "undetermined": "#E0E0E0", # light grey
    },
    "cortex": {
        "upper_layer": "#F44336",  # red
        "deep_layer": "#3F51B5",   # indigo
        "non_neuronal": "#9E9E9E", # grey
        "undetermined": "#E0E0E0", # light grey
    },
    "hsc": {
        "erythroid": "#D32F2F",    # red
        "myeloid": "#1976D2",      # blue
        "lymphoid": "#388E3C",     # green
        "undetermined": "#E0E0E0", # light grey
    },
}

# Draw order: undetermined/non-specific first (background), specific fates on top
DRAW_ORDER = {
    "skin": ["undetermined", "non_appendage", "eccrine", "hair"],
    "pancreas": ["undetermined", "non_endocrine", "delta", "alpha", "beta"],
    "cortex": ["undetermined", "non_neuronal", "deep_layer", "upper_layer"],
    "hsc": ["undetermined", "lymphoid", "myeloid", "erythroid"],
}

UMAP_PARAMS = dict(n_neighbors=30, min_dist=0.3, n_components=2, random_state=42)
POINT_SIZE = 1.0
FIGURE_DPI = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings(system: str):
    """Load adata and all available embedding matrices for a system.

    Returns
    -------
    adata : AnnData
    embeddings : dict[str, np.ndarray]  method_name -> embedding matrix
    """
    paths = SYSTEM_DATA[system]
    print(f"  Loading adata from {paths['adata']} ...")
    adata = ad.read_h5ad(paths["adata"])
    embeddings = {}

    # PRISM
    if "X_prism" in adata.obsm:
        embeddings["PRISM"] = np.array(adata.obsm["X_prism"])
        print(f"    PRISM: {embeddings['PRISM'].shape}")
    else:
        print("    PRISM: not found in obsm, skipping")

    # PCA
    if "X_pca" in adata.obsm:
        pca = np.array(adata.obsm["X_pca"])
        # Use first 50 PCs (or all if fewer)
        embeddings["PCA"] = pca[:, :min(50, pca.shape[1])]
        print(f"    PCA: {embeddings['PCA'].shape}")
    else:
        print("    PCA: not found in obsm, skipping")

    # scGPT
    if os.path.exists(paths["scgpt"]):
        embeddings["scGPT"] = np.load(paths["scgpt"])
        print(f"    scGPT: {embeddings['scGPT'].shape}")
    else:
        print(f"    scGPT: {paths['scgpt']} not found, skipping")

    # Geneformer
    if os.path.exists(paths["geneformer"]):
        embeddings["Geneformer"] = np.load(paths["geneformer"])
        print(f"    Geneformer: {embeddings['Geneformer'].shape}")
    else:
        print(f"    Geneformer: {paths['geneformer']} not found, skipping")

    return adata, embeddings


def compute_umap(emb: np.ndarray) -> np.ndarray:
    """Compute 2-D UMAP from an embedding matrix."""
    reducer = UMAP(**UMAP_PARAMS)
    return reducer.fit_transform(emb)


def plot_umap_panel(ax, coords, labels, system, method, colors, draw_order, show_legend=False):
    """Plot a single UMAP panel on the given axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot in draw order so specific fates appear on top
    for fate in draw_order:
        mask = labels == fate
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors.get(fate, "#BDBDBD"),
            s=POINT_SIZE,
            alpha=0.6,
            rasterized=True,
            linewidths=0,
        )

    ax.set_title(f"{method}", fontsize=10, fontweight="bold", pad=4)

    if show_legend:
        display = FATE_DISPLAY[system]
        handles = []
        for fate in draw_order:
            if fate in display:
                handles.append(
                    mpatches.Patch(color=colors[fate], label=display[fate])
                )
        ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=6,
            frameon=True,
            framealpha=0.8,
            edgecolor="#CCCCCC",
            handlelength=1.0,
            handletextpad=0.4,
            borderpad=0.3,
        )


# ---------------------------------------------------------------------------
# Per-system figures
# ---------------------------------------------------------------------------

def generate_per_system_figure(system, adata, embeddings, umap_coords):
    """Generate a single-system comparison figure (1 row x N columns)."""
    available = [m for m in METHODS if m in umap_coords]
    n = len(available)
    if n == 0:
        print(f"  No embeddings available for {system}, skipping figure.")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8))
    if n == 1:
        axes = [axes]

    labels = adata.obs["fate_label"].values
    colors = FATE_COLORS[system]
    order = DRAW_ORDER[system]

    for i, method in enumerate(available):
        show_legend = (i == n - 1)
        plot_umap_panel(axes[i], umap_coords[method], labels, system, method, colors, order, show_legend=show_legend)

    fig.suptitle(f"{SYSTEM_DISPLAY[system]}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    outpath = f"figures/umap_comparison_{system}.png"
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Combined grid figure
# ---------------------------------------------------------------------------

def generate_grid_figure(all_data):
    """Generate the combined 4-row x 4-col grid figure."""
    n_rows = len(SYSTEMS)
    n_cols = len(METHODS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 15))

    for row, system in enumerate(SYSTEMS):
        if system not in all_data:
            for col in range(n_cols):
                axes[row, col].axis("off")
            continue

        adata, umap_coords = all_data[system]
        labels = adata.obs["fate_label"].values
        colors = FATE_COLORS[system]
        order = DRAW_ORDER[system]

        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            if method not in umap_coords:
                ax.axis("off")
                ax.set_title(f"{method}\n(not available)", fontsize=9, color="#999999")
                continue

            # Show legend on rightmost column
            show_legend = (col == n_cols - 1)
            plot_umap_panel(ax, umap_coords[method], labels, system, method, colors, order, show_legend=show_legend)

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(SYSTEM_DISPLAY[system], fontsize=12, fontweight="bold", labelpad=10)

    # Column headers
    for col, method in enumerate(METHODS):
        axes[0, col].set_title(method, fontsize=12, fontweight="bold", pad=8)

    fig.subplots_adjust(hspace=0.15, wspace=0.08)

    outpath = "figures/umap_comparison_grid.png"
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved combined grid: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    all_data = {}

    for system in SYSTEMS:
        print(f"\n{'='*60}")
        print(f"Processing {SYSTEM_DISPLAY[system]} ...")
        print(f"{'='*60}")

        adata, embeddings = load_embeddings(system)

        # Compute UMAP for each available method
        umap_coords = {}
        for method in METHODS:
            if method not in embeddings:
                continue
            emb = embeddings[method]
            # Handle NaN/Inf
            if np.any(~np.isfinite(emb)):
                print(f"    WARNING: {method} has non-finite values, clipping")
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"    Computing UMAP for {method} ({emb.shape}) ...")
            umap_coords[method] = compute_umap(emb)
            print(f"    Done.")

        # Generate per-system figure
        generate_per_system_figure(system, adata, embeddings, umap_coords)

        # Store for grid
        all_data[system] = (adata, umap_coords)

    # Generate combined grid
    print(f"\n{'='*60}")
    print("Generating combined grid figure ...")
    print(f"{'='*60}")
    generate_grid_figure(all_data)

    print("\nAll done!")


if __name__ == "__main__":
    main()
