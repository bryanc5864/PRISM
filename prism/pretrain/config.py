"""Configuration for PCP pre-training."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PCPConfig:
    # Gene vocabulary
    n_genes: int = 2000
    n_bins: int = 51  # 0=unexpressed, 1-50=expression bins
    mask_token_id: int = 51  # Special MASK bin

    # Model architecture (matches PRISMEncoder for weight transfer)
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 512  # Matches scGPT architecture
    d_output: int = 128  # Contrastive projection output dim
    dropout: float = 0.1
    projection_dims: List[int] = field(default_factory=lambda: [512, 256, 128])

    # scGPT backbone (Stage A)
    scgpt_vocab_size: int = 60697
    scgpt_checkpoint_path: str = "models/scGPT_human/scGPT_human/best_model.pt"
    scgpt_vocab_path: str = "models/scGPT_human/scGPT_human/vocab.json"
    corpus_vocab_path: str = "data/corpus/processed/gene_vocab.json"
    freeze_gene_emb_epochs: int = 2  # Freeze gene embeddings early to preserve scGPT representations

    # Pre-training objectives
    mask_ratio: float = 0.15
    mlm_weight: float = 1.0
    contrastive_weight: float = 1.0

    # Training
    batch_size: int = 256  # Total across GPUs
    n_epochs: int = 10
    lr: float = 1e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    temperature: float = 0.07

    # Batch construction
    perturbations_per_batch: int = 32
    cells_per_perturbation: int = 8

    # Infrastructure
    use_gradient_checkpoint: bool = True
    num_workers: int = 4

    # Paths
    corpus_dir: str = "data/corpus"
    processed_dir: str = "data/corpus/processed"
    checkpoint_dir: str = "checkpoints/pretrain"
    vocab_path: str = "data/corpus/processed/gene_vocab.json"
