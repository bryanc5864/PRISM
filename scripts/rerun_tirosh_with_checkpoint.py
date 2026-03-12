#!/usr/bin/env python3
"""
Re-run Tirosh melanoma pipeline using existing checkpoint.

The marker_scores config was fixed (keys now match fate_categories).
This script:
1. Re-runs data preprocessing to get correct fate labels
2. Loads the existing best checkpoint and extracts PRISM embeddings
3. Runs PRISM-Resolve (Bayesian GMM + Horseshoe DE)
4. Runs PRISM-Trace (trajectory analysis)
5. Runs baseline comparisons
"""

import os
import sys
import time
import json
import numpy as np
import yaml
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Must come before torch import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import anndata as ad

# Import PRISM modules
from prism.config import SystemConfig
from prism.data.preprocess import preprocess_adata, assign_genotypes, assign_labels, split_data, compute_harmony_baseline
from prism.data.dataset import PRISMDataset
from prism.models.encoder import PRISMEncoder
from prism.training.trainer import PRISMTrainer
from prism.utils.metrics import compute_all_metrics

# ---- Configuration ----
SYSTEM_YAML = "configs/tirosh_melanoma.yaml"
DEFAULT_YAML = "configs/default.yaml"
CHECKPOINT_PATH = "checkpoints/tirosh_melanoma/prism_best.pt"

# Load configs
with open(DEFAULT_YAML) as f:
    raw_config = yaml.safe_load(f)
config = {}
for section in raw_config.values():
    if isinstance(section, dict):
        config.update(section)
config["_structured"] = raw_config

system_config = SystemConfig.from_yaml(SYSTEM_YAML)
system_name = system_config.name

# Set per-system directories
config["processed_dir"] = f"data/processed/{system_name}"
config["checkpoint_dir"] = f"checkpoints/{system_name}"
config["figures_dir"] = f"figures/{system_name}"
config["device"] = "cuda:0"

for d in [config["processed_dir"], config["checkpoint_dir"], config["figures_dir"]]:
    os.makedirs(d, exist_ok=True)

device = config["device"]
seed = config.get("seed", 42)
torch.manual_seed(seed)
np.random.seed(seed)

# ============================================================
# STAGE 1: Re-run data preprocessing with fixed config
# ============================================================
print("=" * 60)
print("STAGE 1: Re-run data preprocessing (fixed marker_scores)")
print("=" * 60)

# Check if processed data already exists - we need raw data
raw_dir = config.get("raw_dir", "data/raw")

# Import system-specific download function
import importlib
module_path, func_name = system_config.download_function.rsplit(".", 1)
mod = importlib.import_module(module_path)
download_fn = getattr(mod, func_name)
adata = download_fn(raw_dir=raw_dir)

# Preprocess
forced_genes = system_config.forced_genes or []
adata = preprocess_adata(
    adata,
    min_genes=config.get("min_genes", 200),
    max_genes=getattr(system_config, 'max_genes', None) or config.get("max_genes", 5000),
    max_mito_pct=config.get("max_mito_pct", 5.0),
    n_hvgs=config.get("n_hvgs", 2000),
    forced_genes=forced_genes,
)

# Assign conditions and labels (with fixed marker_scores!)
adata = assign_genotypes(
    adata,
    sample_condition_map=system_config.sample_condition_map or None,
    condition_key=system_config.condition_key,
)
adata = assign_labels(
    adata,
    condition_key=system_config.condition_key,
    conditions=system_config.conditions,
    marker_scores=system_config.marker_scores or None,
    fate_categories=system_config.fate_categories,
    label_strategy=system_config.label_strategy,
    annotation_key=system_config.annotation_key,
    annotation_fate_map=system_config.annotation_fate_map or None,
)

# Harmony
harmony_batch_key = system_config.condition_key
if harmony_batch_key not in adata.obs.columns:
    harmony_batch_key = "sample"
adata = compute_harmony_baseline(adata, batch_key=harmony_batch_key)

# Verify fate labels are now correct
print("\nFate label distribution (should NOT be all non_malignant):")
print(adata.obs["fate_label"].value_counts())
print(f"\nFate int distribution:")
print(adata.obs["fate_int"].value_counts())

n_non_mal = (adata.obs["fate_label"] == "non_malignant").sum()
n_total = len(adata)
if n_non_mal == n_total:
    print("\nERROR: All cells are still non_malignant! Config fix did not work.")
    sys.exit(1)
else:
    print(f"\nFate labels look correct: {n_total - n_non_mal} cells have specific fate labels")

# ============================================================
# STAGE 2: Load checkpoint and extract embeddings
# ============================================================
print("\n" + "=" * 60)
print("STAGE 2: Load checkpoint and extract embeddings")
print("=" * 60)

condition_key = system_config.condition_key
n_fate_categories = len(system_config.fate_categories)

# Determine n_genes and n_conditions from data
n_genes = min(config.get("n_genes", 2000),
              adata.var["highly_variable"].sum() if "highly_variable" in adata.var else 2000)
n_conditions = max(2, adata.obs[condition_key].nunique())

print(f"Shape: {adata.shape}")
print(f"n_genes: {n_genes}, n_conditions: {n_conditions}")

# Build encoder (must match architecture used during training)
encoder = PRISMEncoder(
    n_genes=n_genes,
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
    n_conditions=n_conditions,
    projection_dims=config.get("projection_dims", [512, 256, 128]),
)

# Create trainer and load checkpoint
config["n_fate_categories"] = n_fate_categories
trainer = PRISMTrainer(encoder, config, device=device)

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
# Key is encoder_state_dict (from _save_checkpoint in trainer.py)
state_key = "encoder_state_dict" if "encoder_state_dict" in checkpoint else "model_state_dict"
trainer.encoder.load_state_dict(checkpoint[state_key], strict=False)
trainer.encoder.eval()
print(f"Checkpoint loaded successfully (key={state_key}, epoch={checkpoint.get('epoch')}, val_loss={checkpoint.get('best_val_loss', 'N/A')})")

# Extract embeddings for all cells
batch_size = 32
full_dataset = PRISMDataset(adata, n_genes=n_genes, condition_key=condition_key)
from torch.utils.data import DataLoader
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

full_embeddings, full_labels, full_genotypes = trainer.extract_embeddings(full_loader)
adata.obsm["X_prism"] = full_embeddings
print(f"PRISM embeddings: {full_embeddings.shape}")

# Save updated adata
save_path = os.path.join(config["processed_dir"], "adata_processed.h5ad")
adata.write_h5ad(save_path)
print(f"Saved adata with PRISM embeddings to {save_path}")

# Compute test set metrics
train_adata, val_adata, test_adata = split_data(adata, seed=seed, condition_key=condition_key)
test_dataset = PRISMDataset(test_adata, n_genes=n_genes, condition_key=condition_key)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_embeddings, test_labels, _ = trainer.extract_embeddings(test_loader)

print(f"Split: train={len(train_adata)}, val={len(val_adata)}, test={len(test_adata)}")
test_metrics = compute_all_metrics(test_embeddings, test_labels, method_name="PRISM")
print("\nPRISM Test Metrics:")
for k, v in test_metrics.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v:.4f}")

# ============================================================
# STAGE 3: PRISM-Resolve
# ============================================================
print("\n" + "=" * 60)
print("STAGE 3: PRISM-Resolve (Bayesian DE)")
print("=" * 60)

from prism.resolve.mixture import BayesianFateMixture
from prism.resolve.horseshoe import HorseshoeDE
from prism.utils.visualization import plot_discriminator_genes

embeddings = adata.obsm["X_prism"]
fate_names = system_config.fate_names
known_threshold = system_config.known_fate_threshold

print("\n--- Bayesian Gaussian Mixture ---")
mixture = BayesianFateMixture(n_components=len(fate_names), fate_names=fate_names)

labels = adata.obs["fate_int"].values
label_mask = labels >= known_threshold

mixture.fit(embeddings, labels, label_mask)
fate_probs = mixture.predict_proba(embeddings)
fate_scores = mixture.get_fate_scores(embeddings)
entropy = mixture.compute_entropy(embeddings)

for i, name in enumerate(fate_names):
    if i < fate_probs.shape[1]:
        adata.obs[f"prism_{name}_prob"] = fate_probs[:, i]
adata.obs["prism_fate_entropy"] = entropy

for i, name in enumerate(fate_names):
    count = (fate_probs.argmax(1) == i).sum()
    print(f"  {name}: {count}")

# Horseshoe DE
import scipy.sparse as sp
horseshoe_method = config.get("horseshoe_method", "fast")
print(f"\n--- Horseshoe Differential Expression ({horseshoe_method}) ---")

hvg_mask = adata.var["highly_variable"] if "highly_variable" in adata.var else np.ones(adata.shape[1], dtype=bool)
gene_names = adata.var_names[hvg_mask].tolist()

X = adata[:, hvg_mask].X
if sp.issparse(X):
    X = X.toarray()

de = HorseshoeDE(
    n_warmup=config.get("n_warmup", 2000),
    n_samples=config.get("n_samples", 4000),
    n_chains=config.get("n_chains", 4),
    s0_ratio=config.get("horseshoe_s0_ratio", 0.01),
)

fate_prob_col = fate_probs[:, 1] if fate_probs.shape[1] > 1 else np.random.rand(len(adata))

if horseshoe_method == "mcmc":
    de_results = de.fit_mcmc(X, fate_prob_col, gene_names)
elif horseshoe_method == "full":
    de_results = de.fit(X, fate_prob_col, gene_names)
else:
    de_results = de.fit_fast(X, fate_prob_col, gene_names)

plot_discriminator_genes(de_results, n_top=20, save_path=f"{config['figures_dir']}/discriminator_genes.png")
print(f"Saved discriminator genes plot")

# Top genes
print(f"\n  Genes with PIP > 0.5: {(de_results['posterior_inclusion_prob'] > 0.5).sum()}")
print(f"  Genes with PIP > 0.9: {(de_results['posterior_inclusion_prob'] > 0.9).sum()}")
print(f"\n  Top 10 discriminators:")
print(de_results.head(10)[['gene', 'beta_fate_mean', 'posterior_inclusion_prob']].to_string())

# ============================================================
# STAGE 4: PRISM-Trace
# ============================================================
print("\n" + "=" * 60)
print("STAGE 4: PRISM-Trace (Trajectory)")
print("=" * 60)

from prism.trace.pseudotime import PRISMPseudotime
from prism.trace.branching import BranchAnalyzer

condition_branch_map = system_config.condition_branch_map or {}
branch_names = system_config.branch_names or {"branch_a": "stemlike_branch", "branch_b": "differentiated_branch"}

print("\n--- DPT Pseudotime (PCA space) ---")
pt = PRISMPseudotime(
    n_neighbors=config.get("n_dpt_neighbors", 30),
    n_diffusion_components=config.get("n_diffusion_components", 15),
)
adata = pt.compute(
    adata, embedding_key="X_pca",
    root_cluster=system_config.root_cluster,
    cluster_key=system_config.cluster_key,
    genotype_key=condition_key,
    condition_branch_map=condition_branch_map,
)

pseudotime = adata.obs["dpt_pseudotime"].values
valid_pt = np.isfinite(pseudotime)
print(f"  Valid pseudotime: {valid_pt.sum()}/{len(pseudotime)}")

branch_info = pt.compute_branch_point(adata, fate_probs)
print(f"  Branch point at pseudotime: {branch_info['branch_pseudotime']:.4f}")

branch_a, branch_b = pt.assign_fate_branches(
    adata, fate_probs, percentile_threshold=50,
    fate_names=fate_names, branch_names=branch_names,
)
print(f"  {fate_names[1]} branch: {branch_a.sum()}, {fate_names[2]} branch: {branch_b.sum()}")

# Temporal fate correlation
print("\n--- Temporal Fate Correlation ---")
gene_list = adata.var_names[hvg_mask].tolist()
corr_df = pt.temporal_fate_correlation(adata, fate_probs, gene_list, fdr_threshold=0.05, fate_names=fate_names)
print(f"  Significant genes: {len(corr_df)}")

if not corr_df.empty:
    for direction in corr_df["direction"].unique():
        count = (corr_df["direction"] == direction).sum()
        print(f"  {direction}: {count}")
    print(f"\n  Top 10:")
    print(corr_df.head(10)[['gene', 'spearman_rho', 'direction', 'q_value']].to_string())

# Branch-specific gene programs
print("\n--- Branch-Specific Gene Programs (spline) ---")
branch_analyzer = BranchAnalyzer(
    n_splines=config.get("gam_n_splines", 5),
    fdr_threshold=config.get("branch_fdr", 0.05),
)
branch_df = branch_analyzer.find_branch_genes(
    adata,
    branch_a=branch_names.get("branch_a", "stemlike_branch"),
    branch_b=branch_names.get("branch_b", "differentiated_branch"),
    condition_key=condition_key,
    condition_branch_map=condition_branch_map,
)
print(f"  Spline-divergent genes: {len(branch_df)}")

# Save updated adata
adata.write_h5ad(save_path)

# ============================================================
# STAGE 5: Baselines
# ============================================================
print("\n" + "=" * 60)
print("STAGE 5: Baseline Comparisons")
print("=" * 60)

from prism.experiments.baselines import run_baselines

all_labels = adata.obs["fate_int"].values
baseline_results = run_baselines(adata, all_labels, condition_key=condition_key)

# Add PRISM results
prism_metrics = compute_all_metrics(adata.obsm["X_prism"], all_labels, method_name="PRISM")
baseline_results["PRISM"] = prism_metrics

# Save baseline results
baselines_json_path = os.path.join(config["processed_dir"], "baseline_results.json")
serializable = {}
for method, metrics in baseline_results.items():
    if isinstance(metrics, dict):
        serializable[method] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                 for k, v in metrics.items()}
with open(baselines_json_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"Saved baseline results to {baselines_json_path}")

# Print summary table
print("\n" + "=" * 60)
print("SUMMARY: Tirosh Melanoma Results")
print("=" * 60)
print(f"\n{'Method':<20} {'ARI':>8} {'RF_AUROC':>10} {'RF_F1':>8} {'ASW':>8}")
print("-" * 58)
for method, metrics in baseline_results.items():
    if isinstance(metrics, dict) and "error" not in metrics:
        print(f"{method:<20} {metrics.get('ARI', 0):>8.4f} {metrics.get('RF_AUROC', 0):>10.4f} {metrics.get('RF_F1_macro', 0):>8.4f} {metrics.get('ASW', 0):>8.4f}")

print("\nDONE")
