#!/usr/bin/env python3
"""
PRISM: Progenitor Resolution via Invariance-Sensitive Modeling
Main execution script.

Usage:
    python run_prism.py [--stage STAGE] [--config CONFIG] [--system SYSTEM]

Stages:
    all       - Run complete pipeline
    data      - Download and preprocess data only
    train     - Train PRISM-Encode
    resolve   - Run PRISM-Resolve (Bayesian DE)
    trace     - Run PRISM-Trace (trajectory)
    baselines - Run baseline comparisons
    ablation  - Run ablation studies
    theory    - Run theoretical validations
    benchmark - Run computational benchmarks
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import yaml

# Set environment for GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

# Fix library path for torch
import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Flatten nested config for easy access
    flat = {}
    for section in config.values():
        if isinstance(section, dict):
            flat.update(section)
        else:
            flat[section] = section

    # Keep structured config too
    flat["_structured"] = config
    return flat


def update_results_md(section: str, content: str, results_path: str = "results.md"):
    """Append results to results.md file."""
    with open(results_path, "a") as f:
        f.write(f"\n\n---\n\n### {section}\n")
        f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(content)
        f.write("\n")


def stage_data(config: dict, system_config=None) -> "anndata.AnnData":
    """Stage 1: Download and preprocess data."""
    import anndata as ad
    from prism.data.preprocess import preprocess_adata, assign_genotypes, assign_labels, split_data, compute_harmony_baseline

    print("\n" + "=" * 60)
    print("STAGE: Data Download & Preprocessing")
    print("=" * 60)

    # Download via system-specific download function
    raw_dir = config.get("raw_dir", "data/raw")

    if system_config and system_config.download_function:
        # Dynamic import of download function
        module_path, func_name = system_config.download_function.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        download_fn = getattr(mod, func_name)
        adata = download_fn(raw_dir=raw_dir)
    else:
        from prism.data.download import download_gse220977
        adata = download_gse220977(raw_dir=raw_dir)

    # Get forced genes from system config or training config
    forced_genes = config.get("forced_genes", [])
    if system_config and system_config.forced_genes:
        forced_genes = system_config.forced_genes

    # Preprocess
    adata = preprocess_adata(
        adata,
        min_genes=config.get("min_genes", 200),
        max_genes=getattr(system_config, 'max_genes', None) or config.get("max_genes", 5000),
        max_mito_pct=config.get("max_mito_pct", 5.0),
        n_hvgs=config.get("n_hvgs", 2000),
        forced_genes=forced_genes,
    )

    # Assign conditions/genotypes from sample IDs
    if system_config:
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
    else:
        adata = assign_genotypes(adata)
        adata = assign_labels(adata)

    # Harmony baseline - use condition_key as batch_key, fallback to "sample"
    harmony_batch_key = "sample"
    if system_config and system_config.condition_key:
        harmony_batch_key = system_config.condition_key
    if harmony_batch_key not in adata.obs.columns:
        # Try common alternatives
        for alt_key in ["sample", "batch", "library", "plate"]:
            if alt_key in adata.obs.columns:
                harmony_batch_key = alt_key
                break
    adata = compute_harmony_baseline(adata, batch_key=harmony_batch_key)

    # Save processed
    processed_dir = config.get("processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    save_path = os.path.join(processed_dir, "adata_processed.h5ad")
    adata.write_h5ad(save_path)
    print(f"Saved processed data to {save_path}")

    # Results
    condition_key = system_config.condition_key if system_config else "genotype"
    result_text = f"""
**Dataset**: {system_config.geo_accession if system_config else 'GSE220977'}
- Cells after QC: {adata.shape[0]}
- Genes: {adata.shape[1]}
- HVGs: {adata.var['highly_variable'].sum() if 'highly_variable' in adata.var else 'N/A'}
- Conditions: {dict(adata.obs[condition_key].value_counts()) if condition_key in adata.obs else 'N/A'}
- Fate labels: {dict(adata.obs['fate_label'].value_counts()) if 'fate_label' in adata.obs else 'N/A'}
- Clusters: {dict(adata.obs['cluster'].value_counts()) if 'cluster' in adata.obs else 'N/A'}
"""
    update_results_md("Phase 2: Data Preprocessing", result_text)

    return adata


def stage_train(adata, config: dict, system_config=None):
    """Stage 2: Train PRISM-Encode."""
    import torch
    from prism.data.dataset import PRISMDataset, build_dataloaders
    from prism.data.preprocess import split_data
    from prism.models.encoder import PRISMEncoder
    from prism.training.trainer import PRISMTrainer
    from prism.utils.visualization import plot_training_curves, plot_umap_comparison

    print("\n" + "=" * 60)
    print("STAGE: PRISM-Encode Training")
    print("=" * 60)

    device = config.get("device", "cuda:0")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    condition_key = system_config.condition_key if system_config else "genotype"
    n_fate_categories = len(system_config.fate_categories) if system_config else 4

    # Split data
    train_adata, val_adata, test_adata = split_data(adata, seed=seed, condition_key=condition_key)

    # Create datasets
    n_genes = min(config.get("n_genes", 2000),
                  adata.var["highly_variable"].sum() if "highly_variable" in adata.var else 2000)

    train_dataset = PRISMDataset(train_adata, n_genes=n_genes, condition_key=condition_key)
    val_dataset = PRISMDataset(val_adata, n_genes=n_genes, condition_key=condition_key)
    test_dataset = PRISMDataset(test_adata, n_genes=n_genes, condition_key=condition_key)

    # Build dataloaders — scale batch size to available GPUs
    import torch
    if "batch_size_override" in config:
        batch_size = config["batch_size_override"]
        print(f"  Using overridden batch_size={batch_size}")
    else:
        batch_size = config.get("batch_size", 256)
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if n_gpus < 4:
            batch_size = max(16, 32 * n_gpus)
            print(f"  Adjusted batch_size to {batch_size} for {n_gpus} GPU(s)")
    train_loader, val_loader = build_dataloaders(
        train_dataset, val_dataset,
        batch_size=batch_size, num_workers=0, seed=seed,
    )
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Determine n_conditions from data
    n_conditions = 2  # default
    if condition_key in adata.obs.columns:
        n_conditions = max(2, adata.obs[condition_key].nunique())

    # Build encoder
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

    # Transfer pre-trained weights if checkpoint provided
    pretrained_path = config.get("pretrained_checkpoint")
    if pretrained_path and os.path.exists(pretrained_path):
        import torch
        import json as json_mod
        from prism.pretrain.model import PCPEncoder

        print(f"\nLoading pre-trained weights from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        # Infer PCP d_ff from checkpoint weights
        pcp_d_ff = 512  # default (matches scGPT)
        for k, v in ckpt["encoder_state_dict"].items():
            if "ff.0.weight" in k:  # First FFN linear: (d_ff, d_model)
                pcp_d_ff = v.shape[0]
                break

        # Infer scgpt_vocab_size from checkpoint
        ckpt_config = ckpt.get("config", {})
        pcp_scgpt_vocab_size = ckpt_config.get("scgpt_vocab_size", 60697)

        # Check if checkpoint has universal gene vocab (scGPT-based)
        has_scgpt_vocab = False
        for k, v in ckpt["encoder_state_dict"].items():
            if k == "gene_embedding.weight":
                has_scgpt_vocab = v.shape[0] > n_genes + 2
                break

        # Rebuild encoder with matching architecture
        if pcp_d_ff != config.get("d_ff", 512) or has_scgpt_vocab:
            rebuild_reason = []
            if pcp_d_ff != config.get("d_ff", 512):
                rebuild_reason.append(f"d_ff {config.get('d_ff', 512)} -> {pcp_d_ff}")
            if has_scgpt_vocab:
                rebuild_reason.append(f"gene_vocab_size -> {pcp_scgpt_vocab_size}")
            print(f"  Rebuilding encoder: {', '.join(rebuild_reason)}")
            encoder = PRISMEncoder(
                n_genes=n_genes,
                n_bins=config.get("n_expression_bins", 51),
                d_model=config.get("d_model", 512),
                n_layers=config.get("n_layers", 12),
                n_heads=config.get("n_heads", 8),
                d_ff=pcp_d_ff,
                d_output=config.get("d_output", 256),
                dropout=config.get("dropout", 0.1),
                lora_rank=config.get("lora_rank", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.1),
                n_conditions=n_conditions,
                projection_dims=config.get("projection_dims", [512, 256, 128]),
                gene_vocab_size=pcp_scgpt_vocab_size if has_scgpt_vocab else n_genes,
            )

        # Build PCP encoder with matching architecture
        pcp_encoder = PCPEncoder(
            n_genes=n_genes,
            n_bins=config.get("n_expression_bins", 51),
            d_model=config.get("d_model", 512),
            n_layers=config.get("n_layers", 12),
            n_heads=config.get("n_heads", 8),
            d_ff=pcp_d_ff,
            scgpt_vocab_size=pcp_scgpt_vocab_size if has_scgpt_vocab else n_genes,
        )
        pcp_encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)

        # Build downstream gene_id_map for this system's HVGs
        if has_scgpt_vocab:
            scgpt_vocab_path = config.get("scgpt_vocab_path",
                                           "models/scGPT_human/scGPT_human/vocab.json")
            if os.path.exists(scgpt_vocab_path):
                with open(scgpt_vocab_path) as f:
                    scgpt_vocab = json_mod.load(f)

                # Get HVG names for this system
                hvg_mask = adata.var["highly_variable"] if "highly_variable" in adata.var else None
                if hvg_mask is not None:
                    downstream_genes = adata.var_names[hvg_mask].tolist()[:n_genes]
                else:
                    downstream_genes = adata.var_names[:n_genes].tolist()

                print(f"  Building gene_id_map for {len(downstream_genes)} downstream HVGs")
                pcp_encoder.set_gene_id_map(downstream_genes, scgpt_vocab)

        # Transfer weights to PRISM encoder
        transfer_log = pcp_encoder.transfer_weights_to_prism(encoder)
        n_transferred = sum(1 for v in transfer_log.values() if "transferred" in v)
        n_skipped = sum(1 for v in transfer_log.values() if "skipped" in v)
        n_mismatch = sum(1 for v in transfer_log.values() if "mismatch" in v)
        print(f"  Transferred: {n_transferred}, Skipped: {n_skipped}, Mismatch: {n_mismatch}")
        del pcp_encoder, ckpt

    param_counts = encoder.count_parameters()
    print(f"Encoder parameters: {param_counts}")

    # Train (inject n_fate_categories into config for MINE)
    config["n_fate_categories"] = n_fate_categories
    trainer = PRISMTrainer(encoder, config, device=device)
    n_epochs = config.get("n_epochs", 50)
    patience = config.get("early_stopping_patience", 10)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")

    train_result = trainer.train(
        train_loader, val_loader,
        n_epochs=n_epochs,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
    )

    # Extract embeddings
    embeddings, labels, genotypes = trainer.extract_embeddings(test_loader)

    # Store in adata
    # Create a full DataLoader for all data
    full_dataset = PRISMDataset(adata, n_genes=n_genes, condition_key=condition_key)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    full_embeddings, full_labels, full_genotypes = trainer.extract_embeddings(full_loader)
    adata.obsm["X_prism"] = full_embeddings

    # Save updated adata with embeddings
    processed_dir = config.get("processed_dir", "data/processed")
    adata.write_h5ad(os.path.join(processed_dir, "adata_processed.h5ad"))
    print(f"Saved adata with PRISM embeddings")

    # Plot training curves
    figures_dir = config.get("figures_dir", "figures")
    plot_training_curves(train_result["history"], save_path=f"{figures_dir}/training_curves.png")

    # Plot UMAP comparison
    embeddings_dict = {"PCA": adata.obsm["X_pca"][:, :30]}
    if "X_harmony" in adata.obsm:
        embeddings_dict["Harmony"] = adata.obsm["X_harmony"][:, :30]
    embeddings_dict["PRISM"] = full_embeddings

    # Build label names from system config fate_categories (maps fate_int -> name)
    label_names = None
    if system_config and system_config.fate_categories:
        label_names = {i: name for i, name in enumerate(system_config.fate_categories)}

    plot_umap_comparison(
        embeddings_dict,
        full_labels,
        label_names=label_names,
        save_path=f"{figures_dir}/umap_comparison.png"
    )

    # Compute metrics
    from prism.utils.metrics import compute_all_metrics
    metrics = compute_all_metrics(embeddings, labels, method_name="PRISM")

    # Results
    def _fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    result_text = f"""
**PRISM-Encode Training Results**
- Parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable
- Epochs trained: {train_result['n_epochs_trained']}
- Best validation loss: {train_result['best_val_loss']:.4f}
- Training time: {train_result['total_time_seconds']:.0f}s

**Test Set Metrics**:
- ARI: {_fmt(metrics.get('ARI', 'N/A'))}
- AMI: {_fmt(metrics.get('AMI', 'N/A'))}
- NMI: {_fmt(metrics.get('NMI', 'N/A'))}
- ASW: {_fmt(metrics.get('ASW', 'N/A'))}
- RF F1-macro: {_fmt(metrics.get('RF_F1_macro', 'N/A'))}
- RF AUROC: {_fmt(metrics.get('RF_AUROC', 'N/A'))}
- LR F1-macro: {_fmt(metrics.get('LR_F1_macro', 'N/A'))}
- LR AUROC: {_fmt(metrics.get('LR_AUROC', 'N/A'))}
"""
    update_results_md("Phase 3: PRISM-Encode Training", result_text)

    return adata, trainer, train_dataset, val_dataset, test_dataset, test_loader, labels


def stage_resolve(adata, trainer, config: dict, system_config=None):
    """Stage 3: Run PRISM-Resolve (Bayesian horseshoe DE)."""
    from prism.resolve.mixture import BayesianFateMixture
    from prism.resolve.horseshoe import HorseshoeDE
    from prism.utils.visualization import plot_discriminator_genes

    print("\n" + "=" * 60)
    print("STAGE: PRISM-Resolve (Bayesian DE)")
    print("=" * 60)

    embeddings = adata.obsm["X_prism"]
    fate_names = system_config.fate_names if system_config else ["uncommitted", "eccrine", "hair"]
    known_threshold = system_config.known_fate_threshold if system_config else 2

    # 1. Bayesian GMM for fate assignment
    print("\n--- Bayesian Gaussian Mixture ---")
    mixture = BayesianFateMixture(n_components=len(fate_names), fate_names=fate_names)

    labels = adata.obs["fate_int"].values if "fate_int" in adata.obs else None
    label_mask = labels >= known_threshold if labels is not None else None

    mixture.fit(embeddings, labels, label_mask)
    fate_probs = mixture.predict_proba(embeddings)
    fate_scores = mixture.get_fate_scores(embeddings)
    entropy = mixture.compute_entropy(embeddings)

    # Store fate probabilities with system-specific names
    for i, name in enumerate(fate_names):
        if i < fate_probs.shape[1]:
            adata.obs[f"prism_{name}_prob"] = fate_probs[:, i]
    adata.obs["prism_fate_entropy"] = entropy

    for i, name in enumerate(fate_names):
        count = (fate_probs.argmax(1) == i).sum()
        print(f"  {name}: {count}")

    # 2. Horseshoe DE
    horseshoe_method = config.get("horseshoe_method", "fast")
    print(f"\n--- Horseshoe Differential Expression ({horseshoe_method}) ---")
    import scipy.sparse as sp

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

    # Use the first non-uncommitted fate probability as the discriminator axis
    fate_prob_col = fate_probs[:, 1] if fate_probs.shape[1] > 1 else np.random.rand(len(adata))

    if horseshoe_method == "mcmc":
        de_results = de.fit_mcmc(X, fate_prob_col, gene_names)
    elif horseshoe_method == "full":
        de_results = de.fit(X, fate_prob_col, gene_names)
    else:
        de_results = de.fit_fast(X, fate_prob_col, gene_names)

    # Plot top discriminators
    figures_dir = config.get("figures_dir", "figures")
    plot_discriminator_genes(de_results, n_top=20, save_path=f"{figures_dir}/discriminator_genes.png")

    # Marker recovery
    from prism.utils.metrics import compute_marker_recovery
    ranked_genes = de_results["gene"].tolist()

    known_markers = system_config.known_markers if system_config else {}
    marker_lists = list(known_markers.values()) if known_markers else None
    eccrine_markers = known_markers.get(fate_names[1], None) if len(fate_names) > 1 else None
    hair_markers = known_markers.get(fate_names[2], None) if len(fate_names) > 2 else None
    recovery = compute_marker_recovery(
        ranked_genes,
        known_eccrine_markers=eccrine_markers,
        known_hair_markers=hair_markers,
    )

    # Results
    fate_counts = {name: int((fate_probs.argmax(1) == i).sum()) for i, name in enumerate(fate_names)}
    result_text = f"""
**PRISM-Resolve Results**

**Fate Assignment (Bayesian GMM)**:
"""
    for name, count in fate_counts.items():
        result_text += f"- {name}: {count}\n"
    result_text += f"- Mean fate entropy: {entropy.mean():.4f}\n"
    result_text += f"""
**Horseshoe DE ({horseshoe_method})**:
- Total genes tested: {len(de_results)}
- Genes with PIP > 0.5: {(de_results['posterior_inclusion_prob'] > 0.5).sum()}
- Genes with PIP > 0.9: {(de_results['posterior_inclusion_prob'] > 0.9).sum()}

**Top 10 Cryptic Discriminators**:
{de_results.head(10)[['gene', 'beta_fate_mean', 'posterior_inclusion_prob']].to_string()}

**Marker Gene Recovery**:
"""
    for k in [5, 10, 20]:
        result_text += f"- Precision@{k}: {recovery.get(f'Precision@{k}', 0):.3f}\n"
        result_text += f"- Recall@{k}: {recovery.get(f'Recall@{k}', 0):.3f}\n"

    update_results_md("Phase 4: PRISM-Resolve", result_text)

    return de_results, fate_probs


def stage_trace(adata, fate_probs, config: dict, system_config=None):
    """Stage 4: Run PRISM-Trace (trajectory analysis).

    Uses hybrid approach: DPT in PCA space (connected topology) with
    branch assignment using PRISM fate probabilities.
    """
    from prism.trace.pseudotime import PRISMPseudotime
    from prism.trace.branching import BranchAnalyzer

    print("\n" + "=" * 60)
    print("STAGE: PRISM-Trace (Trajectory)")
    print("=" * 60)

    known_threshold = system_config.known_fate_threshold if system_config else 2
    fate_names = system_config.fate_names if system_config else ["uncommitted", "eccrine", "hair"]
    root_cluster = system_config.root_cluster if system_config else "Epi0"
    cluster_key = system_config.cluster_key if system_config else "cluster"
    condition_key = system_config.condition_key if system_config else "genotype"
    condition_branch_map = system_config.condition_branch_map if system_config else {"WT": "eccrine_branch", "En1-cKO": "hair_branch"}
    branch_names = system_config.branch_names if system_config else {"branch_a": "eccrine_branch", "branch_b": "hair_branch"}

    # If fate_probs not provided, run resolve to get them
    if fate_probs is None:
        print("  Computing fate probabilities from PRISM-Resolve...")
        from prism.resolve.mixture import BayesianFateMixture
        embeddings = adata.obsm["X_prism"]
        labels = adata.obs["fate_int"].values if "fate_int" in adata.obs else None
        label_mask = labels >= known_threshold if labels is not None else None
        mixture = BayesianFateMixture(n_components=len(fate_names), fate_names=fate_names)
        mixture.fit(embeddings, labels, label_mask)
        fate_probs = mixture.predict_proba(embeddings)

    # Pseudotime in PCA space (connected topology)
    print("\n--- DPT Pseudotime (PCA space) ---")
    pt = PRISMPseudotime(
        n_neighbors=config.get("n_dpt_neighbors", 30),
        n_diffusion_components=config.get("n_diffusion_components", 15),
    )
    adata = pt.compute(
        adata, embedding_key="X_pca",
        root_cluster=root_cluster,
        cluster_key=cluster_key,
        genotype_key=condition_key,
        condition_branch_map=condition_branch_map,
    )

    pseudotime = adata.obs["dpt_pseudotime"].values
    valid_pt = np.isfinite(pseudotime)
    print(f"  Valid pseudotime: {valid_pt.sum()}/{len(pseudotime)}")

    # Branch point
    branch_info = pt.compute_branch_point(adata, fate_probs)
    print(f"  Branch point at pseudotime: {branch_info['branch_pseudotime']:.4f}")

    # Fate-based branch assignment
    branch_a, branch_b = pt.assign_fate_branches(
        adata, fate_probs, percentile_threshold=50,
        fate_names=fate_names, branch_names=branch_names,
    )
    print(f"  {fate_names[1] if len(fate_names) > 1 else 'A'} branch: {branch_a.sum()}, "
          f"{fate_names[2] if len(fate_names) > 2 else 'B'} branch: {branch_b.sum()}")

    # Temporal fate correlation (primary analysis)
    print("\n--- Temporal Fate Correlation ---")
    hvg_mask = adata.var["highly_variable"] if "highly_variable" in adata.var else np.ones(adata.shape[1], dtype=bool)
    gene_list = adata.var_names[hvg_mask].tolist()

    corr_df = pt.temporal_fate_correlation(adata, fate_probs, gene_list, fdr_threshold=0.05, fate_names=fate_names)
    print(f"  Significant genes: {len(corr_df)}")

    if not corr_df.empty:
        for direction in corr_df["direction"].unique():
            count = (corr_df["direction"] == direction).sum()
            print(f"  {direction}: {count}")
        print(f"\n  Top 10:")
        print(corr_df.head(10)[["gene", "spearman_rho", "direction", "q_value"]].to_string())

    # Branch-specific genes (spline-based, as secondary analysis)
    print("\n--- Branch-Specific Gene Programs (spline) ---")
    analyzer = BranchAnalyzer(
        n_splines=config.get("gam_n_splines", 5),
        fdr_threshold=config.get("branch_fdr", 0.05),
    )
    branch_df = analyzer.find_branch_genes(
        adata,
        branch_a=branch_names.get("branch_a", "eccrine_branch"),
        branch_b=branch_names.get("branch_b", "hair_branch"),
        condition_key=condition_key,
        condition_branch_map=condition_branch_map,
    )
    cascade = analyzer.build_gene_cascade(branch_df, n_top=20) if not branch_df.empty else branch_df
    print(f"  Spline-divergent genes: {len(branch_df)}")

    # Results
    result_text = f"""
**PRISM-Trace Results**

**Approach**: Hybrid PCA-pseudotime + PRISM-fate branches.
DPT in PCA space (connected topology), branches from PRISM fate probabilities.

**Pseudotime**:
- Valid cells: {valid_pt.sum()}/{len(pseudotime)}
- Branch point pseudotime: {branch_info.get('branch_pseudotime', 'N/A')}
- Committed cells: {branch_info.get('n_committed_cells', 'N/A')}

**Temporal Fate Correlation (FDR < 0.05)**:
- Significant genes: {len(corr_df)}

**Top 10 Temporally Correlated Genes**:
{corr_df.head(10).to_string() if not corr_df.empty else 'None found'}

**Spline-Divergent Genes**: {len(branch_df)}
"""
    update_results_md("Phase 5: PRISM-Trace", result_text)

    return adata


def stage_clonal_validation(adata, config: dict, system_config=None):
    """Stage: Run clonal validation (HSC lineage tracing)."""
    from prism.experiments.clonal_validation import run_clonal_validation

    print("\n" + "=" * 60)
    print("STAGE: Clonal Validation (Lineage Tracing)")
    print("=" * 60)

    if "clone_matrix" not in adata.obsm:
        print("  No clone_matrix in adata.obsm — skipping clonal validation.")
        print("  (Re-run --stage data with force=True to reload with clone matrix)")
        return {}

    embedding_key = "X_prism" if "X_prism" in adata.obsm else "X_pca"
    time_col = "time_point"
    fate_col = "fate_label"

    figures_dir = config.get("figures_dir", "figures")
    results = run_clonal_validation(
        adata,
        fate_col=fate_col,
        time_col=time_col,
        embedding_key=embedding_key,
        save_dir=figures_dir,
    )

    # Write to results.md
    result_text = "**Clonal Validation (Lineage Tracing Ground Truth)**\n\n"

    conc = results.get("concordance", {})
    if "error" not in conc:
        result_text += f"**Clonal Fate Concordance**:\n"
        result_text += f"- Concordance rate: {conc['concordance_rate']:.3f}\n"
        result_text += f"- Tested clones: {conc['n_tested_clones']}\n"
        result_text += f"- Per-fate: {conc.get('per_fate_concordance', {})}\n\n"

    pur = results.get("purity", {})
    if "error" not in pur:
        result_text += f"**Clonal Purity**:\n"
        result_text += f"- Mean purity: {pur['mean_purity']:.3f}\n"
        result_text += f"- Random baseline: {pur['random_baseline']:.3f}\n"
        result_text += f"- Purity over random: {pur['purity_over_random']:.3f}\n\n"

    pred = results.get("predictability", {})
    if "error" not in pred:
        result_text += f"**Fate Predictability (early→late)**:\n"
        result_text += f"- Accuracy: {pred['accuracy']:.3f}\n"
        result_text += f"- F1 macro: {pred['f1_macro']:.3f}\n"
        result_text += f"- AUROC: {pred['auroc']:.3f}\n"
        result_text += f"- Linked early cells: {pred['n_early_cells_linked']}\n"

    update_results_md("Clonal Validation (HSC Lineage Tracing)", result_text)

    return results


def stage_baselines(adata, config: dict, system_config=None):
    """Stage 5: Run baseline comparisons."""
    from prism.experiments.baselines import run_baselines

    print("\n" + "=" * 60)
    print("STAGE: Baseline Comparisons")
    print("=" * 60)

    labels = adata.obs["fate_int"].values if "fate_int" in adata.obs else np.zeros(len(adata))
    condition_key = system_config.condition_key if system_config else "genotype"
    baseline_results = run_baselines(adata, labels, condition_key=condition_key)

    # Add PRISM results if available
    if "X_prism" in adata.obsm:
        from prism.utils.metrics import compute_all_metrics
        prism_metrics = compute_all_metrics(adata.obsm["X_prism"], labels, method_name="PRISM")
        baseline_results["PRISM"] = prism_metrics

    # Save baseline results to JSON for cross-system comparison
    processed_dir = config.get("processed_dir", "data/processed")
    baselines_json_path = os.path.join(processed_dir, "baseline_results.json")
    serializable = {}
    for method, metrics in baseline_results.items():
        if isinstance(metrics, dict):
            serializable[method] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                     for k, v in metrics.items()}
    with open(baselines_json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved baseline results to {baselines_json_path}")

    # Results
    result_text = "**Baseline Comparison (Experiment 1)**\n\n"
    result_text += "| Method | ARI | AMI | NMI | ASW | RF F1 | RF AUROC |\n"
    result_text += "|--------|-----|-----|-----|-----|-------|----------|\n"

    for method, metrics in baseline_results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            result_text += (
                f"| {method} | "
                f"{metrics.get('ARI', 0):.3f} | "
                f"{metrics.get('AMI', 0):.3f} | "
                f"{metrics.get('NMI', 0):.3f} | "
                f"{metrics.get('ASW', 0):.3f} | "
                f"{metrics.get('RF_F1_macro', 0):.3f} | "
                f"{metrics.get('RF_AUROC', 0):.3f} |\n"
            )

    update_results_md("Experiment 1: Cryptic Cell Fate Resolution (Baselines)", result_text)

    return baseline_results


def stage_ablation(train_dataset, val_dataset, test_dataset, test_labels, config):
    """Stage 6: Run ablation studies."""
    from prism.experiments.ablation import run_ablation
    from prism.utils.visualization import plot_ablation_heatmap
    from torch.utils.data import DataLoader

    print("\n" + "=" * 60)
    print("STAGE: Ablation Studies")
    print("=" * 60)

    batch_size = config.get("batch_size", 256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ablation_results = run_ablation(
        train_loader, val_loader, test_loader, test_labels,
        config, device=config.get("device", "cuda:0"),
        n_epochs=config.get("n_epochs", 30),
    )

    # Plot heatmap
    plot_ablation_heatmap(ablation_results, save_path="figures/ablation_heatmap.png")

    # Results
    result_text = "**Ablation Study (Experiment 2)**\n\n"
    result_text += "| Variant | ARI | AMI | RF F1 | RF AUROC | Time(s) |\n"
    result_text += "|---------|-----|-----|-------|----------|------|\n"

    for variant, metrics in ablation_results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            result_text += (
                f"| {variant} | "
                f"{metrics.get('ARI', 0):.3f} | "
                f"{metrics.get('AMI', 0):.3f} | "
                f"{metrics.get('RF_F1_macro', 0):.3f} | "
                f"{metrics.get('RF_AUROC', 0):.3f} | "
                f"{metrics.get('training_time', 0):.0f} |\n"
            )

    update_results_md("Experiment 2: Ablation Studies", result_text)

    return ablation_results


def stage_theory(config: dict):
    """Stage 7: Run theoretical validations."""
    from prism.experiments.theory import run_theory_validation

    print("\n" + "=" * 60)
    print("STAGE: Theoretical Validations")
    print("=" * 60)

    theory_results = run_theory_validation(
        n_cells=config.get("n_cells_theory", 5000),
        n_genes=config.get("n_genes", 2000),
        seed=config.get("seed", 42),
    )

    # Results
    result_text = "**Theoretical Validation Results**\n\n"

    for theorem, results in theory_results.items():
        result_text += f"\n**{theorem}**:\n"
        if isinstance(results, dict):
            verified_key = [k for k in results.keys() if "verified" in k]
            if verified_key:
                result_text += f"- Verified: {results[verified_key[0]]}\n"
            for k, v in results.items():
                if isinstance(v, (int, float, bool)):
                    result_text += f"- {k}: {v}\n"

    update_results_md("Theoretical Validations", result_text)

    return theory_results


def stage_benchmarks(dataset, config: dict):
    """Stage 8: Run computational benchmarks."""
    from prism.experiments.benchmarks import run_benchmarks

    print("\n" + "=" * 60)
    print("STAGE: Computational Benchmarks")
    print("=" * 60)

    bench_results = run_benchmarks(
        dataset, config,
        device=config.get("device", "cuda:0"),
    )

    result_text = "**Computational Benchmarking (Experiment 8)**\n\n"

    if "training" in bench_results:
        t = bench_results["training"]
        result_text += f"**Training**:\n"
        result_text += f"- Time per epoch: {t.get('time_per_epoch_seconds', 0):.1f}s\n"
        result_text += f"- Estimated 50 epochs: {t.get('estimated_50_epochs_minutes', 0):.1f} min\n"

    if "inference" in bench_results:
        i = bench_results["inference"]
        result_text += f"\n**Inference**:\n"
        result_text += f"- Time per cell: {i.get('time_per_cell_ms', 0):.3f} ms\n"
        result_text += f"- Throughput: {i.get('throughput_cells_per_second', 0):.0f} cells/s\n"

    if "memory" in bench_results:
        m = bench_results["memory"]
        result_text += f"\n**Memory**:\n"
        result_text += f"- Model: {m.get('model_memory_gb', 0):.2f} GB\n"
        result_text += f"- Peak: {m.get('peak_memory_gb', 0):.2f} GB\n"

    update_results_md("Experiment 8: Computational Benchmarking", result_text)

    return bench_results


def main():
    parser = argparse.ArgumentParser(description="PRISM Pipeline")
    parser.add_argument("--stage", default="all", help="Pipeline stage to run")
    parser.add_argument("--config", default="configs/default.yaml", help="Training config file path")
    parser.add_argument("--system", default=None, help="System config file path (e.g., configs/skin.yaml)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (skip auto-scaling)")
    parser.add_argument("--pretrained", default=None, help="Path to PCP pre-trained checkpoint for weight transfer")
    args = parser.parse_args()

    # Load training config
    config = load_config(args.config)
    config["device"] = args.device
    if args.batch_size is not None:
        config["batch_size_override"] = args.batch_size
    if args.pretrained:
        config["pretrained_checkpoint"] = args.pretrained

    # Load system config
    system_config = None
    if args.system:
        from prism.config import SystemConfig
        system_config = SystemConfig.from_yaml(args.system)
        print(f"System: {system_config.name}")
    else:
        # Default to skin config for backward compatibility
        from prism.config import SKIN_CONFIG
        system_config = SKIN_CONFIG

    # Per-system output directories (skin uses default paths for backward compat)
    system_name = system_config.name if system_config else "skin"
    if system_name != "skin":
        config["processed_dir"] = f"data/processed/{system_name}"
        config["checkpoint_dir"] = f"checkpoints/{system_name}"
        config["figures_dir"] = f"figures/{system_name}"
    else:
        config.setdefault("processed_dir", "data/processed")
        config.setdefault("checkpoint_dir", "checkpoints")
        config.setdefault("figures_dir", "figures")

    os.makedirs(config["processed_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config.get("figures_dir", "figures"), exist_ok=True)

    print("=" * 60)
    print("PRISM: Progenitor Resolution via Invariance-Sensitive Modeling")
    print("=" * 60)
    print(f"Stage: {args.stage}")
    print(f"System: {system_config.name}")
    print(f"Device: {config['device']}")
    print(f"Config: {args.config}")
    print(f"Output: {config['processed_dir']}")

    start_time = time.time()

    if args.stage in ["all", "data"]:
        adata = stage_data(config, system_config)

    if args.stage in ["all", "train"]:
        if "adata" not in locals():
            import anndata as ad
            processed_path = os.path.join(
                config.get("processed_dir", "data/processed"),
                "adata_processed.h5ad"
            )
            if os.path.exists(processed_path):
                adata = ad.read_h5ad(processed_path)
            else:
                adata = stage_data(config, system_config)

        adata, trainer, train_ds, val_ds, test_ds, test_loader, test_labels = stage_train(adata, config, system_config)

    if args.stage in ["all", "resolve"]:
        if "adata" not in locals():
            import anndata as ad
            adata = ad.read_h5ad(os.path.join(config.get("processed_dir", "data/processed"), "adata_processed.h5ad"))
        de_results, fate_probs = stage_resolve(adata, locals().get("trainer"), config, system_config)

    if args.stage in ["all", "trace"]:
        if "adata" not in locals():
            import anndata as ad
            adata = ad.read_h5ad(os.path.join(config.get("processed_dir", "data/processed"), "adata_processed.h5ad"))
        adata = stage_trace(adata, locals().get("fate_probs"), config, system_config)

    # Clonal validation (HSC-specific, or any system with clone_matrix)
    if args.stage in ["all", "clonal_validation"]:
        if "adata" not in locals():
            import anndata as ad
            adata = ad.read_h5ad(os.path.join(config.get("processed_dir", "data/processed"), "adata_processed.h5ad"))
        clonal_results = stage_clonal_validation(adata, config, system_config)

    # Experiment stages (baselines, ablation, theory, benchmarks)
    is_skin = system_name == "skin"

    if args.stage in ["all", "baselines"]:
        if "adata" not in locals():
            import anndata as ad
            adata = ad.read_h5ad(os.path.join(config.get("processed_dir", "data/processed"), "adata_processed.h5ad"))
        baseline_results = stage_baselines(adata, config, system_config)

    if args.stage in (["all"] if is_skin else []) + ["ablation"]:
        if "train_ds" in locals():
            ablation_results = stage_ablation(train_ds, val_ds, test_ds, test_labels, config)

    if args.stage in (["all"] if is_skin else []) + ["theory"]:
        theory_results = stage_theory(config)

    if args.stage in (["all"] if is_skin else []) + ["benchmark"]:
        if "train_ds" in locals():
            bench_results = stage_benchmarks(train_ds, config)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PRISM pipeline complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")

    update_results_md("Pipeline Complete", f"Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
