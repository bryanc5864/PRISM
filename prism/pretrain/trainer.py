"""PCP Trainer: Multi-GPU pre-training with contrastive + MLM objectives."""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, Optional

from .model import PCPEncoder
from .config import PCPConfig


def supervised_contrastive_loss(z, labels, temperature=0.07):
    """SupCon loss: cells with same perturbation are positive pairs.

    Args:
        z: (B, D) L2-normalized embeddings
        labels: (B,) perturbation IDs

    Returns:
        loss: scalar
    """
    B = z.shape[0]
    device = z.device

    # Similarity matrix
    sim = z @ z.T / temperature  # (B, B)

    # Positive mask: same perturbation label
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = pos_mask & self_mask  # Exclude self-pairs

    # Check if any positives exist
    n_positives = pos_mask.sum()
    if n_positives == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Log-sum-exp for numerical stability
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max) * self_mask.float()

    # Log probability of each pair
    log_prob = (sim - sim_max) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-prob over positive pairs
    loss = -(log_prob * pos_mask.float()).sum() / n_positives.float()
    return loss


class PCPTrainer:
    """Pre-training pipeline with multi-GPU DataParallel.

    Multi-task loss:
        L = contrastive_weight * L_supcon + mlm_weight * L_mlm
    """

    def __init__(self, encoder: PCPEncoder, config: PCPConfig, device: str = "cuda:0"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Model
        self.encoder = encoder.to(self.device)
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.n_gpus > 1:
            gpu_ids = list(range(self.n_gpus))
            self.encoder_dp = nn.DataParallel(self.encoder, device_ids=gpu_ids)
            print(f"Multi-GPU: {self.n_gpus} GPUs {gpu_ids}")
        else:
            self.encoder_dp = self.encoder
            print(f"Single-GPU: {self.device}")

        # Optimizer
        self.optimizer = AdamW(
            self.encoder.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # LR scheduler with linear warmup + cosine decay
        total_steps = config.n_epochs * 30000  # Estimate, updated after first epoch
        self.scheduler = self._build_scheduler(total_steps)

        # AMP
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Training state
        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.history = []
        self._pending_scheduler_state = None

    def _build_scheduler(self, total_steps: int):
        """Linear warmup + cosine decay."""
        warmup = self.config.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train one epoch."""
        self.encoder.train()

        # Gene embedding freeze schedule: preserve scGPT representations in early epochs
        if epoch < self.config.freeze_gene_emb_epochs:
            self.encoder.gene_embedding.weight.requires_grad = False
            if epoch == 0:
                print(f"  Gene embeddings FROZEN (epoch {epoch} < {self.config.freeze_gene_emb_epochs})")
        else:
            self.encoder.gene_embedding.weight.requires_grad = True
            if epoch == self.config.freeze_gene_emb_epochs:
                print(f"  Gene embeddings UNFROZEN (epoch {epoch})")

        total_loss = 0
        total_contrastive = 0
        total_mlm = 0
        total_mlm_acc = 0
        n_batches = 0
        nan_batches = 0

        for batch in dataloader:
            expression = batch["expression"].to(self.device)
            masked_expression = batch["masked_expression"].to(self.device)
            mask = batch["mask"].to(self.device)
            pert_ids = batch["perturbation_id"].to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.use_amp):
                # Forward pass (DataParallel handles splitting)
                z, mlm_logits, cls_repr = self.encoder_dp(
                    expression, masked_expression, mask
                )

                # 1. Supervised contrastive loss
                loss_contrastive = supervised_contrastive_loss(
                    z, pert_ids, self.config.temperature
                )

                # 2. Masked gene prediction loss (select masked positions)
                # Cast to fp32 before cross-entropy to prevent fp16 overflow
                # mlm_logits: (B, n_genes, n_bins+1), mask: (B, n_genes)
                masked_logits = mlm_logits[mask].float()  # (M, n_bins+1)

                # Clamp logits to prevent fp16 overflow in cross_entropy
                masked_logits = masked_logits.clamp(-100, 100)

                mlm_targets = expression[mask].long().clamp(0, self.config.n_bins)
                loss_mlm = F.cross_entropy(masked_logits, mlm_targets)

                # Combined loss
                loss = (
                    self.config.contrastive_weight * loss_contrastive
                    + self.config.mlm_weight * loss_mlm
                )

            # Skip NaN/Inf batches instead of poisoning the entire epoch
            if not torch.isfinite(loss):
                nan_batches += 1
                self.scheduler.step()
                self.global_step += 1
                if nan_batches % 10 == 1:
                    print(
                        f"  Step {self.global_step} | SKIPPED NaN/Inf batch "
                        f"(total: {nan_batches}) | "
                        f"Con: {loss_contrastive.item():.4f} | "
                        f"MLM: {loss_mlm.item():.4f}",
                        flush=True,
                    )
                continue

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            # Metrics
            total_loss += loss.item()
            total_contrastive += loss_contrastive.item()
            total_mlm += loss_mlm.item()
            with torch.no_grad():
                mlm_pred = masked_logits.argmax(dim=-1)
                total_mlm_acc += (mlm_pred == mlm_targets).float().mean().item()
            n_batches += 1

            # Print every 100 steps
            if self.global_step % 100 == 0:
                print(
                    f"  Step {self.global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Con: {loss_contrastive.item():.4f} | "
                    f"MLM: {loss_mlm.item():.4f} | "
                    f"MLM-Acc: {(mlm_pred == mlm_targets).float().mean():.3f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        if nan_batches > 0:
            print(f"  Epoch {epoch}: {nan_batches} NaN batches skipped out of "
                  f"{n_batches + nan_batches} total", flush=True)

        return {
            "epoch": epoch,
            "loss": total_loss / max(n_batches, 1),
            "contrastive_loss": total_contrastive / max(n_batches, 1),
            "mlm_loss": total_mlm / max(n_batches, 1),
            "mlm_accuracy": total_mlm_acc / max(n_batches, 1),
            "lr": self.scheduler.get_last_lr()[0],
            "global_step": self.global_step,
            "nan_batches": nan_batches,
        }

    def train(
        self,
        dataloader: DataLoader,
        n_epochs: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict:
        """Full training loop."""
        n_epochs = n_epochs or self.config.n_epochs
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Rebuild scheduler with correct total_steps
        steps_per_epoch = len(dataloader)
        total_steps = n_epochs * steps_per_epoch
        self.scheduler = self._build_scheduler(total_steps)

        # Restore scheduler position from checkpoint if available
        if hasattr(self, '_pending_scheduler_state') and self._pending_scheduler_state is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None

        params = self.encoder.count_parameters()
        print(f"\nPCP Pre-training: {n_epochs} epochs, {steps_per_epoch} steps/epoch")
        print(f"Parameters: {params['total']:,} total ({params['trainable']:,} trainable)")
        print(f"Batch size: {self.config.perturbations_per_batch * self.config.cells_per_perturbation}")
        print(f"AMP: {self.use_amp}, GPUs: {self.n_gpus}")
        print()

        start_time = time.time()

        for epoch in range(self.start_epoch, n_epochs):
            self.current_epoch = epoch
            # Reseed sampler for reproducible shuffling each epoch
            if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'set_epoch'):
                dataloader.batch_sampler.set_epoch(epoch)
            epoch_start = time.time()
            metrics = self.train_epoch(dataloader, epoch)
            epoch_time = time.time() - epoch_start

            self.history.append(metrics)

            print(
                f"Epoch {epoch}/{n_epochs} ({epoch_time:.0f}s) | "
                f"Loss: {metrics['loss']:.4f} | "
                f"SupCon: {metrics['contrastive_loss']:.4f} | "
                f"MLM: {metrics['mlm_loss']:.4f} | "
                f"MLM-Acc: {metrics['mlm_accuracy']:.3f}"
            )

            # Save checkpoint
            if metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                self._save_checkpoint(checkpoint_dir, "best")

            # Save periodic checkpoint
            if (epoch + 1) % 2 == 0 or epoch == n_epochs - 1:
                self._save_checkpoint(checkpoint_dir, f"epoch_{epoch}")

        # Save final
        self._save_checkpoint(checkpoint_dir, "final")

        total_time = time.time() - start_time
        print(f"\nPre-training complete: {total_time:.0f}s ({total_time/3600:.1f}h)")
        print(f"Best loss: {self.best_loss:.4f}")

        # Save training history
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return {"history": self.history, "best_loss": self.best_loss, "total_time": total_time}

    def load_checkpoint(self, ckpt: Dict):
        """Restore full training state from checkpoint dict."""
        self.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("  Warning: No optimizer state in checkpoint, starting fresh optimizer")
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.start_epoch = ckpt.get("epoch", 0) + 1

        # Store scheduler state dict for deferred loading in train()
        # (train() rebuilds scheduler with correct total_steps, then we load state)
        self._pending_scheduler_state = ckpt.get("scheduler_state_dict", None)

        # Restore GradScaler state if available
        if "scaler_state_dict" in ckpt and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        print(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}, "
              f"best_loss={self.best_loss:.4f}")

    def _save_checkpoint(self, checkpoint_dir: str, tag: str):
        path = os.path.join(checkpoint_dir, f"pcp_{tag}.pt")
        # Unwrap DataParallel if needed
        encoder_state = (
            self.encoder.state_dict()
            if not isinstance(self.encoder_dp, nn.DataParallel)
            else self.encoder.state_dict()
        )
        save_dict = {
            "encoder_state_dict": encoder_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": getattr(self, 'current_epoch', 0),
            "best_loss": self.best_loss,
            "config": {
                "n_genes": self.config.n_genes,
                "n_bins": self.config.n_bins,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "d_ff": self.config.d_ff,
                "d_output": self.config.d_output,
                "scgpt_vocab_size": self.config.scgpt_vocab_size,
            },
        }
        if self.scaler is not None:
            save_dict["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(save_dict, path)
