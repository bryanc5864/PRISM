"""
Main training loop for PRISM.

Multi-GPU training using nn.DataParallel:
- Encoder is wrapped with DataParallel to split batches across all GPUs
- Contrastive loss + MINE run on primary GPU after gathering embeddings
  (they need the full batch for proper pair mining)
- Gradient checkpointing in transformer blocks reduces per-GPU memory

Three-stage training:
1. Domain-adaptive pretraining (optional)
2. Contrastive fine-tuning with hard-negative curriculum
3. Bayesian inference (PRISM-Resolve)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import json

from torch.amp import autocast, GradScaler

from ..models.encoder import PRISMEncoder
from ..models.contrastive import HardNegativeInfoNCE, CurriculumScheduler, compute_raw_similarity_matrix
from ..models.mine import MINEEstimator
from .curriculum import HardNegativeCurriculum


class PRISMTrainer:
    """Main PRISM training pipeline with multi-GPU support.

    Uses nn.DataParallel to distribute the encoder forward pass across
    all available GPUs. The contrastive loss and MINE estimator run on
    the primary GPU since they require the full gathered batch for
    proper positive/negative pair construction.
    """

    def __init__(
        self,
        encoder: PRISMEncoder,
        config: dict,
        device: str = "cuda:0",
    ):
        self.config = config
        self.primary_device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Detect available GPUs
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_multi_gpu = self.n_gpus > 1

        # Model — wrap with DataParallel if multiple GPUs
        self.encoder = encoder.to(self.primary_device)
        if self.use_multi_gpu:
            gpu_ids = list(range(self.n_gpus))
            self.encoder_dp = nn.DataParallel(self.encoder, device_ids=gpu_ids)
            print(f"Multi-GPU training enabled: {self.n_gpus} GPUs {gpu_ids}")
        else:
            self.encoder_dp = self.encoder
            print(f"Single-GPU training on {self.primary_device}")

        # Loss functions — stay on primary device (need full batch)
        self.contrastive_loss = HardNegativeInfoNCE(
            temperature_init=config.get("temperature_init", 0.07),
        ).to(self.primary_device)

        # MINE estimator for information-preserving regularizer
        self.mine = MINEEstimator(
            embedding_dim=config.get("projection_dims", [512, 256, 128])[-1],
            n_labels=config.get("n_fate_categories", 4),
        ).to(self.primary_device)

        # Curriculum scheduler
        self.curriculum = HardNegativeCurriculum(
            alpha_max=config.get("alpha_max", 2.0),
            warmup_epochs=config.get("curriculum_warmup_epochs", 10),
        )

        # Optimizer with separate param groups
        param_groups = self.encoder.get_all_trainable_params()
        lr_lora = config.get("lr_lora", 2e-4)
        lr_head = config.get("lr_head", 1e-3)

        optimizer_params = []
        for group in param_groups:
            lr = lr_lora if group["lr_group"] == "lora" else lr_head
            optimizer_params.append({
                "params": group["params"],
                "lr": lr,
            })

        # Add MINE and contrastive loss params
        optimizer_params.append({
            "params": self.mine.parameters(),
            "lr": lr_head,
        })
        optimizer_params.append({
            "params": self.contrastive_loss.parameters(),
            "lr": lr_head,
        })

        self.optimizer = AdamW(
            optimizer_params,
            weight_decay=config.get("weight_decay", 0.01),
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        )

        # LR scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("n_epochs", 50),
            eta_min=config.get("lr_min_lora", 1e-6),
        )

        # AMP (Automatic Mixed Precision) for faster training
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict:
        """Train for one epoch across all GPUs."""
        self.encoder.train()
        self.mine.train()

        # Update curriculum
        alpha = self.curriculum.get_alpha(epoch)
        self.contrastive_loss.alpha = alpha

        total_loss = 0
        total_contrastive = 0
        total_mine = 0
        total_recon = 0
        n_batches = 0

        lambda_info = self.config.get("info_reg_lambda", 0.1)
        mu_recon = self.config.get("recon_weight", 0.1)
        grad_clip = self.config.get("gradient_clip", 1.0)

        for batch in train_loader:
            expression = batch["expression"].to(self.primary_device)
            raw_expr = batch["raw_expression"].to(self.primary_device)
            genotype = batch["genotype"].to(self.primary_device)
            fate_label = batch["fate_label"].to(self.primary_device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with autocast("cuda", enabled=self.use_amp):
                # DataParallel splits across GPUs, gathers on primary
                z, cls_repr, recon = self.encoder_dp(
                    expression, genotype,
                    return_reconstruction=True,
                )

                # 1. Contrastive loss with hard-negative weighting
                raw_sim = compute_raw_similarity_matrix(raw_expr)
                loss_contrastive, contrastive_metrics = self.contrastive_loss(
                    z, fate_label, raw_sim, genotype
                )

                # 2. Information-preserving regularizer (MINE)
                loss_mine, mine_metrics = self.mine.compute_regularizer(
                    z.detach(), fate_label, lambda_info=lambda_info,
                )
                mi_through_encoder, _ = self.mine.compute_regularizer(
                    z, fate_label, lambda_info=lambda_info
                )

                # 3. Reconstruction loss
                mask = torch.rand_like(raw_expr) < 0.15
                recon_target = raw_expr[mask]
                recon_pred = recon[mask]
                loss_recon = F.mse_loss(recon_pred, recon_target)

                # Combined loss
                loss = loss_contrastive + mi_through_encoder + mu_recon * loss_recon

            # Backward pass with AMP scaler
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(self.mine.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(self.mine.parameters(), grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            total_contrastive += loss_contrastive.item()
            total_mine += mine_metrics.get("mi_estimate", 0)
            total_recon += loss_recon.item()
            n_batches += 1

        self.scheduler.step()

        metrics = {
            "epoch": epoch,
            "alpha": alpha,
            "loss": total_loss / max(n_batches, 1),
            "contrastive_loss": total_contrastive / max(n_batches, 1),
            "mine_mi": total_mine / max(n_batches, 1),
            "recon_loss": total_recon / max(n_batches, 1),
            "lr": self.optimizer.param_groups[0]["lr"],
            "temperature": self.contrastive_loss.temperature.item(),
        }

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict:
        """Validate on held-out data."""
        self.encoder.eval()
        self.mine.eval()

        total_loss = 0
        total_contrastive = 0
        n_batches = 0

        for batch in val_loader:
            expression = batch["expression"].to(self.primary_device)
            raw_expr = batch["raw_expression"].to(self.primary_device)
            genotype = batch["genotype"].to(self.primary_device)
            fate_label = batch["fate_label"].to(self.primary_device)

            with autocast("cuda", enabled=self.use_amp):
                z, cls_repr, _ = self.encoder_dp(expression, genotype)
                raw_sim = compute_raw_similarity_matrix(raw_expr)
                loss_contrastive, _ = self.contrastive_loss(
                    z, fate_label, raw_sim, genotype
                )

            total_loss += loss_contrastive.item()
            total_contrastive += loss_contrastive.item()
            n_batches += 1

        metrics = {
            "val_loss": total_loss / max(n_batches, 1),
            "val_contrastive_loss": total_contrastive / max(n_batches, 1),
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        patience: int = 10,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict:
        """Full training loop with early stopping.

        Returns:
            Training history dict
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        gpu_info = f"{self.n_gpus} GPUs (DataParallel)" if self.use_multi_gpu else str(self.primary_device)
        print(f"Starting PRISM training: {n_epochs} epochs, {gpu_info}")
        param_counts = self.encoder.count_parameters()
        print(f"Parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")

        start_time = time.time()

        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            self.training_history.append(metrics)

            # Print progress
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:3d}/{n_epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Val: {metrics['val_loss']:.4f} | "
                    f"α: {metrics['alpha']:.2f} | "
                    f"τ: {metrics['temperature']:.4f} | "
                    f"MI: {metrics['mine_mi']:.3f} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Early stopping
            if metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = metrics["val_loss"]
                self.patience_counter = 0
                self._save_checkpoint(checkpoint_dir, "best")
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        # Save final checkpoint
        self._save_checkpoint(checkpoint_dir, "final")

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f}min)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return {
            "history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "total_time_seconds": total_time,
            "n_epochs_trained": epoch + 1,
        }

    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract PRISM embeddings for all cells.

        Returns:
            embeddings: (N, d) L2-normalized embeddings
            fate_labels: (N,) fate labels
            genotypes: (N,) genotype labels
        """
        self.encoder.eval()

        all_z = []
        all_cls = []
        all_labels = []
        all_genotypes = []

        for batch in dataloader:
            expression = batch["expression"].to(self.primary_device)
            genotype = batch["genotype"].to(self.primary_device)

            with autocast("cuda", enabled=self.use_amp):
                z, cls_repr, _ = self.encoder_dp(expression, genotype)

            all_z.append(z.cpu().numpy())
            all_cls.append(cls_repr.cpu().numpy())
            all_labels.append(batch["fate_label"].numpy())
            all_genotypes.append(batch["genotype"].numpy())

        embeddings = np.concatenate(all_z, axis=0)
        cls_embeddings = np.concatenate(all_cls, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        genotypes = np.concatenate(all_genotypes, axis=0)

        return embeddings, labels, genotypes

    def _save_checkpoint(self, checkpoint_dir: str, tag: str):
        """Save model checkpoint (unwrap DataParallel)."""
        path = os.path.join(checkpoint_dir, f"prism_{tag}.pt")
        # Save the underlying module, not the DataParallel wrapper
        encoder_state = self.encoder.state_dict()
        torch.save({
            "encoder_state_dict": encoder_state,
            "contrastive_loss_state_dict": self.contrastive_loss.state_dict(),
            "mine_state_dict": self.mine.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.primary_device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.contrastive_loss.load_state_dict(checkpoint["contrastive_loss_state_dict"])
        self.mine.load_state_dict(checkpoint["mine_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from epoch {self.epoch}, val_loss={self.best_val_loss:.4f}")
