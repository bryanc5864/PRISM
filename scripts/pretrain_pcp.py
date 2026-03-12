#!/usr/bin/env python
"""Launch PCP (Perturbation-Contrastive Pre-training) on corpus.

Supports two initialization modes:
- Random init (original): --no-init-scgpt
- scGPT backbone (Stage A → B): --init-scgpt (default)

Usage:
    # Stage A+B: Initialize from scGPT, train on perturbation corpus
    python scripts/pretrain_pcp.py --init-scgpt

    # Stage B only: Random init (original behavior)
    python scripts/pretrain_pcp.py --no-init-scgpt

    # Resume training
    python scripts/pretrain_pcp.py --resume checkpoints/pretrain/pcp_epoch_4.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from prism.pretrain.config import PCPConfig
from prism.pretrain.model import PCPEncoder
from prism.pretrain.dataset import build_pretraining_dataloader
from prism.pretrain.trainer import PCPTrainer


def main():
    parser = argparse.ArgumentParser(description="PCP Pre-training")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--perts-per-batch", type=int, default=32)
    parser.add_argument("--cells-per-pert", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--processed-dir", type=str, default="data/corpus/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--warmup-steps", type=int, default=2000)

    # scGPT initialization (Stage A)
    parser.add_argument("--init-scgpt", action="store_true", default=True,
                        help="Initialize from scGPT pre-trained weights (default: True)")
    parser.add_argument("--no-init-scgpt", action="store_false", dest="init_scgpt",
                        help="Use random initialization instead of scGPT")
    parser.add_argument("--scgpt-checkpoint", type=str,
                        default="models/scGPT_human/scGPT_human/best_model.pt",
                        help="Path to scGPT best_model.pt")
    parser.add_argument("--scgpt-vocab", type=str,
                        default="models/scGPT_human/scGPT_human/vocab.json",
                        help="Path to scGPT vocab.json")
    parser.add_argument("--corpus-vocab", type=str,
                        default="data/corpus/processed/gene_vocab.json",
                        help="Path to corpus gene_vocab.json")
    parser.add_argument("--scgpt-vocab-size", type=int, default=60697,
                        help="scGPT vocabulary size")
    parser.add_argument("--freeze-gene-emb-epochs", type=int, default=2,
                        help="Freeze gene embeddings for N epochs to preserve scGPT representations")
    args = parser.parse_args()

    # Build config
    config = PCPConfig(
        n_genes=args.n_genes,
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_epochs=args.n_epochs,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        perturbations_per_batch=args.perts_per_batch,
        cells_per_perturbation=args.cells_per_pert,
        warmup_steps=args.warmup_steps,
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        scgpt_vocab_size=args.scgpt_vocab_size,
        scgpt_checkpoint_path=args.scgpt_checkpoint,
        scgpt_vocab_path=args.scgpt_vocab,
        corpus_vocab_path=args.corpus_vocab,
        freeze_gene_emb_epochs=args.freeze_gene_emb_epochs if args.init_scgpt else 0,
    )

    print("=" * 60)
    print("  PCP Pre-training")
    print("=" * 60)
    print(f"  Model: {config.n_layers}L/{config.d_model}d/{config.n_heads}H")
    print(f"  FFN dim: {config.d_ff}")
    print(f"  Genes: {config.n_genes}, Bins: {config.n_bins}")
    print(f"  Gene vocab: {'scGPT (' + str(config.scgpt_vocab_size) + ')' if args.init_scgpt else 'positional (' + str(config.n_genes) + ')'}")
    print(f"  Batch: {config.perturbations_per_batch}P × {config.cells_per_perturbation}K "
          f"= {config.perturbations_per_batch * config.cells_per_perturbation}")
    print(f"  Epochs: {config.n_epochs}, LR: {config.lr}")
    print(f"  Mask ratio: {config.mask_ratio}")
    print(f"  Init: {'scGPT backbone' if args.init_scgpt else 'random'}")
    if args.init_scgpt:
        print(f"  Freeze gene emb: {config.freeze_gene_emb_epochs} epochs")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print()

    # Build dataloader
    print("Loading preprocessed corpus...")
    dataloader = build_pretraining_dataloader(
        processed_dir=config.processed_dir,
        config=config,
    )

    # Build model
    scgpt_vocab_size = config.scgpt_vocab_size if args.init_scgpt else config.n_genes
    encoder = PCPEncoder(
        n_genes=config.n_genes,
        n_bins=config.n_bins,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        d_output=config.d_output,
        dropout=config.dropout,
        projection_dims=config.projection_dims,
        use_gradient_checkpoint=config.use_gradient_checkpoint,
        scgpt_vocab_size=scgpt_vocab_size,
    )

    # Initialize from scGPT (Stage A)
    if args.init_scgpt and not args.resume:
        print("\n--- Stage A: Loading scGPT pre-trained weights ---")

        if not os.path.exists(args.scgpt_checkpoint):
            print(f"ERROR: scGPT checkpoint not found: {args.scgpt_checkpoint}")
            print("Download from: https://github.com/bowang-lab/scGPT")
            sys.exit(1)

        # Load corpus gene names for ID mapping
        corpus_gene_names = None
        if os.path.exists(args.corpus_vocab):
            with open(args.corpus_vocab) as f:
                corpus_vocab = json.load(f)
            corpus_gene_names = corpus_vocab.get("genes", list(corpus_vocab.get("gene_to_idx", {}).keys()))
            print(f"  Corpus genes: {len(corpus_gene_names)}")

        # Transfer scGPT weights
        transfer_log = PCPEncoder.load_scgpt_weights(
            encoder,
            scgpt_checkpoint_path=args.scgpt_checkpoint,
            scgpt_vocab_path=args.scgpt_vocab,
            corpus_gene_names=corpus_gene_names,
        )
        print()

    # Build trainer
    trainer = PCPTrainer(encoder, config)

    # Resume if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        trainer.load_checkpoint(ckpt)

    # Train (Stage B)
    print("\n--- Stage B: Perturbation-Contrastive Pre-training ---")
    results = trainer.train(
        dataloader=dataloader,
        n_epochs=config.n_epochs,
        checkpoint_dir=config.checkpoint_dir,
    )

    print(f"\nDone. Best loss: {results['best_loss']:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}/")


if __name__ == "__main__":
    main()
