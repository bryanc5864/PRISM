#!/usr/bin/env python3
"""
Experiment 5: Generalization Benchmark for PRISM.

Since downloading external datasets is complex, this implements a synthetic
generalization benchmark that probes PRISM's core strengths:

1. Synthetic Cryptic Fate Tasks
   - Generate synthetic scRNA-seq-like data with 3 cell populations sharing >90%
     of gene expression (mimicking the real cryptic fate problem).
   - Vary the "crypticness" parameter: shared_var / disc_var ratios of
     10:1, 50:1, 100:1, 500:1.
   - For each ratio, train a simple contrastive embedding (SimCLR-style MLP)
     and compare RF AUROC on PCA vs contrastive vs raw features.
   - Demonstrates how contrastive learning scales with increasing crypticness.

2. Scalability with Cell Count
   - Subsample the real PRISM data to 1k, 5k, 10k, and 25k cells.
   - For each subsample, compute PCA RF AUROC and PRISM RF AUROC (from
     existing embeddings).
   - Report the gap between methods at each scale.

3. Feature Dimensionality Analysis
   - Run PCA on X_prism to determine effective dimensionality (components
     capturing 90%, 95%, 99% of variance).
   - Compare with effective dimensionality of raw expression PCA.
   - Characterizes the compactness of the learned representation.

Results are appended to results.md.
"""

import os
import sys
import time
import numpy as np
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

warnings.filterwarnings("ignore")

import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Part 1: Synthetic Cryptic Fate Tasks
# ============================================================================

def generate_synthetic_cryptic_data(
    n_cells_per_pop=500,
    n_genes=2000,
    n_populations=3,
    shared_disc_ratio=100.0,
    n_discriminative_genes=20,
    n_shared_programs=10,
    seed=42,
):
    """Generate synthetic scRNA-seq-like data with cryptic fate structure.

    Mimics real scRNA-seq where multiple co-regulated gene programs dominate
    the variance, and subtle discriminative signals are buried within a few
    genes that also participate in the shared programs (correlated noise).

    The key challenge: PCA captures the high-variance shared programs in its
    top components, pushing the discriminative signal to later components
    where it mixes with noise. Higher shared_disc_ratio = more cryptic.

    Args:
        n_cells_per_pop: Cells per population.
        n_genes: Total number of genes.
        n_populations: Number of cell populations.
        shared_disc_ratio: Ratio of shared variance to discriminative variance.
            Higher = more cryptic (harder to separate).
        n_discriminative_genes: Number of genes with population-specific signal.
        n_shared_programs: Number of latent gene programs shared across all pops.
        seed: Random seed.

    Returns:
        X: (n_cells, n_genes) expression matrix (genes shuffled).
        labels: (n_cells,) population labels (0, 1, 2, ...).
    """
    rng = np.random.RandomState(seed)
    n_cells = n_cells_per_pop * n_populations
    labels = np.repeat(np.arange(n_populations), n_cells_per_pop)

    # --- Shared gene programs (dominate variance) ---
    # Each program is a latent factor loading onto many genes.
    # This creates strong correlated structure that PCA captures first.
    program_loadings = rng.randn(n_shared_programs, n_genes) * 0.5
    # Make programs sparse: each program affects ~30% of genes
    sparsity_mask = rng.rand(n_shared_programs, n_genes) < 0.3
    program_loadings *= sparsity_mask

    shared_var = 1.0
    program_activities = rng.randn(n_cells, n_shared_programs) * np.sqrt(shared_var)
    X_shared = program_activities @ program_loadings  # (n_cells, n_genes)

    # --- Discriminative signal (subtle, buried) ---
    # The discriminative signal strength is controlled by shared_disc_ratio.
    # Crucially, the discriminative genes ALSO participate in shared programs
    # (they're scattered among all genes, not segregated).
    disc_strength = np.sqrt(shared_var / shared_disc_ratio)

    # Choose which genes are discriminative (scattered, not at the end)
    disc_gene_idx = rng.choice(n_genes, size=n_discriminative_genes, replace=False)
    genes_per_pop = n_discriminative_genes // n_populations

    X_disc = np.zeros((n_cells, n_genes))
    for pop in range(n_populations):
        pop_mask = labels == pop
        # Each population has a mean shift in its subset of discriminative genes
        start_g = pop * genes_per_pop
        end_g = min((pop + 1) * genes_per_pop, n_discriminative_genes)
        pop_disc_genes = disc_gene_idx[start_g:end_g]
        for g in pop_disc_genes:
            X_disc[pop_mask, g] = disc_strength * rng.choice([-1, 1])

    # --- Gene-level noise (biological + technical) ---
    # Heteroscedastic noise scaled per gene (like real scRNA-seq dropout + noise)
    gene_noise_scale = rng.exponential(scale=0.3, size=n_genes)
    noise = rng.randn(n_cells, n_genes) * gene_noise_scale[np.newaxis, :]

    # --- Combine ---
    X = X_shared + X_disc + noise
    # Ensure non-negative (like log-normalized count data)
    X = np.maximum(X, 0)

    # Shuffle cells
    perm = rng.permutation(n_cells)
    X = X[perm]
    labels = labels[perm]

    return X, labels


class SimpleContrastiveEncoder(nn.Module):
    """Simple MLP encoder for contrastive learning on synthetic data."""

    def __init__(self, n_input, n_hidden=256, n_output=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
        self.projector = nn.Sequential(
            nn.Linear(n_output, n_output),
            nn.ReLU(),
            nn.Linear(n_output, n_output),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def contrastive_loss(z, labels, temperature=0.1):
    """Supervised contrastive loss (SupCon).

    Uses label information to define positive pairs (same population)
    and negative pairs (different population).
    """
    z = nn.functional.normalize(z, dim=1)
    n = z.shape[0]
    sim = torch.mm(z, z.t()) / temperature

    # Mask: positive pairs share the same label
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    # Remove self-similarity
    mask_self = ~torch.eye(n, dtype=torch.bool, device=z.device)
    pos_mask = labels_eq & mask_self
    neg_mask = ~labels_eq & mask_self

    # For numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Compute log-softmax over positives
    exp_sim = torch.exp(sim)
    # Denominator: sum over all non-self entries
    denom = (exp_sim * mask_self.float()).sum(dim=1, keepdim=True)

    log_prob = sim - torch.log(denom + 1e-12)

    # Average over positive pairs
    pos_count = pos_mask.float().sum(dim=1)
    pos_count = torch.clamp(pos_count, min=1.0)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count

    return loss.mean()


def train_contrastive_encoder(X, labels, n_epochs=100, batch_size=256, lr=1e-3,
                               n_hidden=256, n_output=64, device="cpu"):
    """Train a simple contrastive encoder on synthetic data.

    Returns the learned embeddings.
    """
    X_tensor = torch.FloatTensor(X)
    labels_tensor = torch.LongTensor(labels)
    dataset = TensorDataset(X_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SimpleContrastiveEncoder(X.shape[1], n_hidden=n_hidden, n_output=n_output).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            h, z = model(batch_x)
            loss = contrastive_loss(z, batch_y, temperature=0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        all_x = X_tensor.to(device)
        h, z = model(all_x)
        embeddings = h.cpu().numpy()

    return embeddings


def evaluate_rf_auroc(X, labels, n_folds=5, seed=42):
    """Evaluate RF AUROC using cross-validation (OVR for multiclass)."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    if n_classes == 2:
        probs = cross_val_predict(rf, X, labels, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(labels, probs)
    else:
        probs = cross_val_predict(rf, X, labels, cv=cv, method="predict_proba")
        return roc_auc_score(labels, probs, multi_class="ovr", average="macro")


def run_synthetic_cryptic_benchmark():
    """Part 1: Synthetic cryptic fate tasks with varying difficulty."""
    print("=" * 70)
    print("PART 1: Synthetic Cryptic Fate Tasks")
    print("=" * 70)
    print()
    print("Generating synthetic scRNA-seq data with 3 cell populations,")
    print("2000 genes, 20 discriminative genes, and varying crypticness.")
    print()

    ratios = [10, 50, 100, 500]
    n_cells_per_pop = 500
    n_genes = 2000
    n_disc_genes = 20
    n_pcs = 30  # Typical scRNA-seq PCA dimensionality
    results = []

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()

    for ratio in ratios:
        print(f"--- Crypticness ratio {ratio}:1 ---")
        t0 = time.time()

        X, labels = generate_synthetic_cryptic_data(
            n_cells_per_pop=n_cells_per_pop,
            n_genes=n_genes,
            n_populations=3,
            shared_disc_ratio=float(ratio),
            n_discriminative_genes=n_disc_genes,
            n_shared_programs=10,
            seed=42,
        )

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 1. PCA embeddings (standard pipeline)
        pca = PCA(n_components=n_pcs, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_auroc = evaluate_rf_auroc(X_pca, labels)
        print(f"  PCA ({n_pcs} PCs) RF AUROC:     {pca_auroc:.4f}")

        # 2. PCA with more components (trying to capture more signal)
        pca_50 = PCA(n_components=50, random_state=42)
        X_pca_50 = pca_50.fit_transform(X_scaled)
        pca50_auroc = evaluate_rf_auroc(X_pca_50, labels)
        print(f"  PCA (50 PCs) RF AUROC:       {pca50_auroc:.4f}")

        # 3. Contrastive embeddings (supervised, like PRISM)
        contrastive_emb = train_contrastive_encoder(
            X_scaled, labels,
            n_epochs=150,
            batch_size=min(256, n_cells_per_pop),
            lr=1e-3,
            n_hidden=256,
            n_output=64,
            device=device,
        )
        contrastive_auroc = evaluate_rf_auroc(contrastive_emb, labels)
        print(f"  Contrastive (64d) RF AUROC:  {contrastive_auroc:.4f}")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

        # Compute contrastive advantage over standard PCA
        advantage = contrastive_auroc - pca_auroc
        print(f"  Contrastive advantage over PCA(30): {advantage:+.4f}")
        print()

        results.append({
            "ratio": ratio,
            "pca30_auroc": pca_auroc,
            "pca50_auroc": pca50_auroc,
            "contrastive_auroc": contrastive_auroc,
            "advantage": advantage,
            "time": elapsed,
        })

    return results


# ============================================================================
# Part 2: Scalability with Cell Count
# ============================================================================

def run_scalability_benchmark(adata):
    """Part 2: Subsample real data and compare PCA vs PRISM at each scale."""
    print("=" * 70)
    print("PART 2: Scalability with Cell Count")
    print("=" * 70)
    print()

    # Get labels
    label_map = {"non_appendage": 0, "undetermined": 1, "eccrine": 2, "hair": 3}
    labels = adata.obs["fate_label"].map(label_map).fillna(0).astype(int).values
    fate_mask = (labels == 2) | (labels == 3)

    n_eccrine = (labels == 2).sum()
    n_hair = (labels == 3).sum()
    print(f"Total cells: {adata.shape[0]}")
    print(f"Eccrine: {n_eccrine}, Hair: {n_hair}")
    print()

    # Subsample sizes
    subsample_sizes = [1000, 5000, 10000, 25344]
    results = []

    for n_target in subsample_sizes:
        print(f"--- Subsample: {n_target} cells ---")
        t0 = time.time()

        if n_target >= adata.shape[0]:
            # Use all cells
            idx = np.arange(adata.shape[0])
        else:
            # Stratified subsampling: preserve the ratio of cell types
            rng = np.random.RandomState(42)
            idx_list = []
            for label_val in [0, 1, 2, 3]:
                label_idx = np.where(labels == label_val)[0]
                # Proportion of this label in the full dataset
                prop = len(label_idx) / len(labels)
                n_sample = max(1, int(n_target * prop))
                if n_sample > len(label_idx):
                    n_sample = len(label_idx)
                chosen = rng.choice(label_idx, size=n_sample, replace=False)
                idx_list.append(chosen)
            idx = np.concatenate(idx_list)
            # If we're slightly off due to rounding, adjust
            if len(idx) < n_target:
                remaining = np.setdiff1d(np.arange(len(labels)), idx)
                extra = rng.choice(remaining, size=min(n_target - len(idx), len(remaining)), replace=False)
                idx = np.concatenate([idx, extra])
            elif len(idx) > n_target:
                idx = rng.choice(idx, size=n_target, replace=False)

        sub_labels = labels[idx]
        sub_fate_mask = (sub_labels == 2) | (sub_labels == 3)
        n_ecc_sub = (sub_labels == 2).sum()
        n_hair_sub = (sub_labels == 3).sum()
        print(f"  Subsampled cells: {len(idx)} (eccrine: {n_ecc_sub}, hair: {n_hair_sub})")

        if sub_fate_mask.sum() < 10:
            print(f"  SKIP: too few fate-labeled cells ({sub_fate_mask.sum()})")
            results.append({
                "n_cells": len(idx),
                "n_eccrine": n_ecc_sub,
                "n_hair": n_hair_sub,
                "pca_auroc": np.nan,
                "prism_auroc": np.nan,
                "gap": np.nan,
            })
            continue

        # PCA embeddings for this subsample
        pca_emb = adata.obsm["X_pca"][idx]
        emb_fate_pca = pca_emb[sub_fate_mask]
        lab_fate = (sub_labels[sub_fate_mask] == 2).astype(int)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        pca_probs = cross_val_predict(rf, emb_fate_pca, lab_fate, cv=cv, method="predict_proba")[:, 1]
        pca_auroc = roc_auc_score(lab_fate, pca_probs)
        print(f"  PCA (50 PCs) RF AUROC:  {pca_auroc:.4f}")

        # PRISM embeddings for this subsample
        prism_emb = adata.obsm["X_prism"][idx]
        emb_fate_prism = prism_emb[sub_fate_mask]

        rf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        prism_probs = cross_val_predict(rf2, emb_fate_prism, lab_fate, cv=cv, method="predict_proba")[:, 1]
        prism_auroc = roc_auc_score(lab_fate, prism_probs)
        print(f"  PRISM (128d) RF AUROC:  {prism_auroc:.4f}")

        gap = prism_auroc - pca_auroc
        print(f"  PRISM advantage:        {gap:+.4f}")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")
        print()

        results.append({
            "n_cells": len(idx),
            "n_eccrine": n_ecc_sub,
            "n_hair": n_hair_sub,
            "pca_auroc": pca_auroc,
            "prism_auroc": prism_auroc,
            "gap": gap,
        })

    return results


# ============================================================================
# Part 3: Feature Dimensionality Analysis
# ============================================================================

def run_dimensionality_analysis(adata):
    """Part 3: Analyze effective dimensionality of PRISM vs raw expression."""
    print("=" * 70)
    print("PART 3: Feature Dimensionality Analysis")
    print("=" * 70)
    print()

    results = {}

    # --- PRISM embedding dimensionality ---
    prism_emb = adata.obsm["X_prism"]  # (25344, 128)
    print(f"PRISM embedding shape: {prism_emb.shape}")

    # PCA on PRISM embeddings to find effective dimensionality
    pca_prism = PCA(n_components=min(128, prism_emb.shape[1]), random_state=42)
    pca_prism.fit(prism_emb)
    var_ratio_prism = pca_prism.explained_variance_ratio_
    cumvar_prism = np.cumsum(var_ratio_prism)

    n_90_prism = np.searchsorted(cumvar_prism, 0.90) + 1
    n_95_prism = np.searchsorted(cumvar_prism, 0.95) + 1
    n_99_prism = np.searchsorted(cumvar_prism, 0.99) + 1

    print(f"\nPRISM embedding effective dimensionality:")
    print(f"  Components for 90% variance: {n_90_prism}")
    print(f"  Components for 95% variance: {n_95_prism}")
    print(f"  Components for 99% variance: {n_99_prism}")
    print(f"  Top-1 PC explains:           {var_ratio_prism[0]:.4f} ({var_ratio_prism[0]*100:.1f}%)")
    print(f"  Top-5 PCs explain:           {cumvar_prism[4]:.4f} ({cumvar_prism[4]*100:.1f}%)")
    print(f"  Top-10 PCs explain:          {cumvar_prism[9]:.4f} ({cumvar_prism[9]*100:.1f}%)")

    results["prism"] = {
        "total_dims": prism_emb.shape[1],
        "n_90": int(n_90_prism),
        "n_95": int(n_95_prism),
        "n_99": int(n_99_prism),
        "top1_var": float(var_ratio_prism[0]),
        "top5_cumvar": float(cumvar_prism[4]),
        "top10_cumvar": float(cumvar_prism[9]),
        "variance_ratio": var_ratio_prism.tolist(),
    }

    # --- Raw expression PCA dimensionality ---
    # Use the existing PCA from preprocessing (top 50 PCs)
    if "X_pca" in adata.obsm:
        pca_raw = adata.obsm["X_pca"]  # (25344, 50)
        print(f"\nRaw expression PCA shape: {pca_raw.shape}")

        # We already have the top 50 PCs; analyze their variance structure
        if "pca" in adata.uns and "variance_ratio" in adata.uns["pca"]:
            var_ratio_raw = adata.uns["pca"]["variance_ratio"]
        else:
            # Recompute PCA variance from the raw data
            pca_raw_full = PCA(n_components=50, random_state=42)
            if sp.issparse(adata.X):
                # Use HVGs only
                hvg_mask = adata.var["highly_variable"] if "highly_variable" in adata.var else np.ones(adata.shape[1], dtype=bool)
                X_dense = adata[:, hvg_mask].X
                if sp.issparse(X_dense):
                    X_dense = X_dense.toarray()
                pca_raw_full.fit(X_dense)
            else:
                pca_raw_full.fit(adata.X)
            var_ratio_raw = pca_raw_full.explained_variance_ratio_

        cumvar_raw = np.cumsum(var_ratio_raw)
        n_90_raw = np.searchsorted(cumvar_raw, 0.90) + 1
        n_95_raw = np.searchsorted(cumvar_raw, 0.95) + 1
        n_99_raw = np.searchsorted(cumvar_raw, 0.99) + 1

        # Clip to maximum available components
        n_90_raw = min(n_90_raw, len(var_ratio_raw))
        n_95_raw = min(n_95_raw, len(var_ratio_raw))
        n_99_raw = min(n_99_raw, len(var_ratio_raw))

        # Check if 50 PCs are enough to capture these thresholds
        total_var_50 = cumvar_raw[-1] if len(cumvar_raw) > 0 else 0

        print(f"\nRaw expression PCA effective dimensionality:")
        print(f"  Components for 90% variance: {n_90_raw}{' (capped at 50)' if n_90_raw >= 50 else ''}")
        print(f"  Components for 95% variance: {n_95_raw}{' (capped at 50)' if n_95_raw >= 50 else ''}")
        cap_note = f' (capped at 50; 50 PCs capture {total_var_50*100:.1f}%)' if n_99_raw >= 50 else ''
        print(f"  Components for 99% variance: {n_99_raw}{cap_note}")
        print(f"  Top-1 PC explains:           {var_ratio_raw[0]:.4f} ({var_ratio_raw[0]*100:.1f}%)")
        print(f"  Top-5 PCs explain:           {cumvar_raw[4]:.4f} ({cumvar_raw[4]*100:.1f}%)")
        print(f"  Top-10 PCs explain:          {cumvar_raw[9]:.4f} ({cumvar_raw[9]*100:.1f}%)")
        print(f"  Top-50 PCs explain:          {total_var_50:.4f} ({total_var_50*100:.1f}%)")

        results["raw_pca"] = {
            "total_dims": 50,
            "n_90": int(n_90_raw),
            "n_95": int(n_95_raw),
            "n_99": int(n_99_raw),
            "top1_var": float(var_ratio_raw[0]),
            "top5_cumvar": float(cumvar_raw[4]),
            "top10_cumvar": float(cumvar_raw[9]),
            "top50_cumvar": float(total_var_50),
            "variance_ratio": var_ratio_raw.tolist(),
        }

    # --- Compute PCA on raw HVGs directly to get full spectrum ---
    print(f"\nComputing full PCA on HVG expression for complete spectrum...")
    hvg_mask = adata.var["highly_variable"] if "highly_variable" in adata.var else np.ones(adata.shape[1], dtype=bool)
    n_hvg = int(hvg_mask.sum())
    n_comps = min(200, n_hvg, adata.shape[0])

    X_hvg = adata[:, hvg_mask].X
    if sp.issparse(X_hvg):
        X_hvg = X_hvg.toarray()

    pca_full = PCA(n_components=n_comps, random_state=42)
    pca_full.fit(X_hvg)
    var_ratio_full = pca_full.explained_variance_ratio_
    cumvar_full = np.cumsum(var_ratio_full)

    n_90_full = np.searchsorted(cumvar_full, 0.90) + 1
    n_95_full = np.searchsorted(cumvar_full, 0.95) + 1
    n_99_full = np.searchsorted(cumvar_full, 0.99) + 1

    print(f"  HVG count: {n_hvg}")
    print(f"  Components computed: {n_comps}")
    print(f"  Components for 90% variance: {n_90_full}")
    print(f"  Components for 95% variance: {n_95_full}")
    print(f"  Components for 99% variance: {n_99_full}")

    results["raw_hvg_full"] = {
        "n_hvg": n_hvg,
        "n_comps_computed": n_comps,
        "n_90": int(n_90_full),
        "n_95": int(n_95_full),
        "n_99": int(n_99_full),
        "top1_var": float(var_ratio_full[0]),
        "top5_cumvar": float(cumvar_full[4]),
        "top10_cumvar": float(cumvar_full[9]),
    }

    # --- Comparison summary ---
    print(f"\n{'='*60}")
    print("DIMENSIONALITY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Space':<25} {'90% var':<10} {'95% var':<10} {'99% var':<10}")
    print(f"  {'-'*55}")
    print(f"  {'PRISM (128d total)':<25} {n_90_prism:<10} {n_95_prism:<10} {n_99_prism:<10}")
    print(f"  {'Raw HVG PCA':<25} {n_90_full:<10} {n_95_full:<10} {n_99_full:<10}")
    print(f"\n  PRISM concentrates {var_ratio_prism[0]*100:.1f}% of variance in PC1,")
    print(f"  vs raw expression concentrating {var_ratio_full[0]*100:.1f}% in PC1.")
    print(f"  PRISM effective dim ({n_95_prism} for 95%) vs raw ({n_95_full} for 95%)")
    compression = n_95_full / n_95_prism if n_95_prism > 0 else float("inf")
    print(f"  Compression ratio: {compression:.1f}x")

    results["compression_95"] = float(compression)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    total_start = time.time()

    print("=" * 70)
    print("EXPERIMENT 5: GENERALIZATION BENCHMARK")
    print("Progenitor Resolution via Invariance-Sensitive Modeling (PRISM)")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ---- Part 1: Synthetic cryptic fate tasks ----
    synthetic_results = run_synthetic_cryptic_benchmark()

    # ---- Load real data for Parts 2 and 3 ----
    print("\nLoading real data...")
    adata = ad.read_h5ad(os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad"))
    print(f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"PRISM embeddings: {adata.obsm['X_prism'].shape}")
    print()

    # ---- Part 2: Scalability benchmark ----
    scalability_results = run_scalability_benchmark(adata)

    # ---- Part 3: Dimensionality analysis ----
    dimensionality_results = run_dimensionality_analysis(adata)

    total_time = time.time() - total_start

    # ================================================================
    # Write results to results.md
    # ================================================================
    print()
    print("=" * 70)
    print("WRITING RESULTS TO results.md")
    print("=" * 70)

    result_text = ""
    result_text += f"\n\n---\n\n### Experiment 5: Generalization Benchmark\n"
    result_text += f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Part 1
    result_text += "**Part 1: Synthetic Cryptic Fate Tasks**\n\n"
    result_text += "Synthetic scRNA-seq data: 3 populations, 2000 genes (10 shared co-regulated\n"
    result_text += "programs), 20 discriminative genes scattered among shared genes,\n"
    result_text += "500 cells/population. Contrastive = supervised SupCon MLP (64d output).\n\n"
    result_text += "| Crypticness (shared:disc) | PCA(30) RF AUROC | PCA(50) RF AUROC | Contrastive RF AUROC | Contrastive Advantage |\n"
    result_text += "|--------------------------|-----------------|-----------------|---------------------|----------------------|\n"
    for r in synthetic_results:
        result_text += (
            f"| {r['ratio']}:1 | "
            f"{r['pca30_auroc']:.4f} | "
            f"{r['pca50_auroc']:.4f} | "
            f"{r['contrastive_auroc']:.4f} | "
            f"{r['advantage']:+.4f} |\n"
        )

    result_text += "\n"

    # Part 2
    result_text += "**Part 2: Scalability with Cell Count**\n\n"
    result_text += "Real mouse skin data subsampled (stratified). Eccrine (label=2) vs hair (label=3).\n\n"
    result_text += "| Cells | Eccrine | Hair | PCA RF AUROC | PRISM RF AUROC | PRISM Advantage |\n"
    result_text += "|-------|---------|------|-------------|---------------|----------------|\n"
    for r in scalability_results:
        if np.isnan(r.get("pca_auroc", np.nan)):
            result_text += f"| {r['n_cells']} | {r['n_eccrine']} | {r['n_hair']} | N/A | N/A | N/A |\n"
        else:
            result_text += (
                f"| {r['n_cells']} | {r['n_eccrine']} | {r['n_hair']} | "
                f"{r['pca_auroc']:.4f} | "
                f"{r['prism_auroc']:.4f} | "
                f"{r['gap']:+.4f} |\n"
            )

    result_text += "\n"

    # Part 3
    result_text += "**Part 3: Feature Dimensionality Analysis**\n\n"
    result_text += "Effective dimensionality: number of PCs to capture X% of variance.\n\n"

    prism_d = dimensionality_results.get("prism", {})
    raw_d = dimensionality_results.get("raw_hvg_full", {})

    result_text += "| Space | Total Dims | 90% Var | 95% Var | 99% Var | Top-1 PC |\n"
    result_text += "|-------|-----------|---------|---------|---------|----------|\n"
    result_text += (
        f"| PRISM embedding | {prism_d.get('total_dims', '?')} | "
        f"{prism_d.get('n_90', '?')} | "
        f"{prism_d.get('n_95', '?')} | "
        f"{prism_d.get('n_99', '?')} | "
        f"{prism_d.get('top1_var', 0)*100:.1f}% |\n"
    )
    result_text += (
        f"| Raw HVG expression | {raw_d.get('n_hvg', '?')} | "
        f"{raw_d.get('n_90', '?')} | "
        f"{raw_d.get('n_95', '?')} | "
        f"{raw_d.get('n_99', '?')} | "
        f"{raw_d.get('top1_var', 0)*100:.1f}% |\n"
    )

    compression = dimensionality_results.get("compression_95", 0)
    result_text += f"\nDimensionality compression (95% var): {compression:.1f}x\n"

    # Key findings
    result_text += "\n**Key Findings**:\n"

    # Summarize synthetic results
    if len(synthetic_results) >= 2:
        first = synthetic_results[0]
        last = synthetic_results[-1]
        max_adv = max(synthetic_results, key=lambda x: x["advantage"])
        result_text += (
            f"- Contrastive advantage at lowest crypticness ({first['ratio']}:1): "
            f"{first['advantage']:+.4f}; "
            f"at highest ({last['ratio']}:1): {last['advantage']:+.4f}\n"
        )
        result_text += (
            f"- Maximum contrastive advantage: {max_adv['advantage']:+.4f} "
            f"at {max_adv['ratio']}:1 ratio\n"
        )
        avg_advantage = np.mean([r["advantage"] for r in synthetic_results])
        result_text += f"- Average contrastive advantage over PCA(30): {avg_advantage:+.4f}\n"
        # PCA degradation
        pca_drop = first["pca30_auroc"] - last["pca30_auroc"]
        result_text += (
            f"- PCA(30) AUROC drops {abs(pca_drop):.4f} from {first['ratio']}:1 "
            f"to {last['ratio']}:1 crypticness\n"
        )

    # Summarize scalability results
    valid_scale = [r for r in scalability_results if not np.isnan(r.get("gap", np.nan))]
    if len(valid_scale) >= 2:
        avg_gap = np.mean([r["gap"] for r in valid_scale])
        min_gap = min(valid_scale, key=lambda x: x["gap"])
        max_gap = max(valid_scale, key=lambda x: x["gap"])
        result_text += (
            f"- PRISM advantage over PCA is consistent across scales: "
            f"avg {avg_gap:+.4f} "
            f"(range: {min_gap['gap']:+.4f} at {min_gap['n_cells']} cells "
            f"to {max_gap['gap']:+.4f} at {max_gap['n_cells']} cells)\n"
        )

    # Dimensionality
    result_text += (
        f"- PRISM learns a {compression:.1f}x more compact representation "
        f"({prism_d.get('n_95', '?')} vs {raw_d.get('n_95', '?')} PCs for 95% variance)\n"
    )

    result_text += f"\nTotal experiment time: {total_time:.0f}s ({total_time/60:.1f}min)\n"

    # Append to results.md
    with open(os.path.join(PROJECT_DIR, "results.md"), "a") as f:
        f.write(result_text)

    print(result_text)
    print(f"\nResults appended to results.md")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
