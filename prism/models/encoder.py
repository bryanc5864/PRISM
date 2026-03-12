"""
PRISM-Encode: Transformer encoder with LoRA adaptation.

Architecture aligned with scGPT/PCP for weight transfer:
- Universal gene vocabulary (scGPT 60,697 tokens + CLS + PAD)
- Gene ID mapping via registered buffer
- Post-norm transformer blocks (matching scGPT)
- Flash attention via scaled_dot_product_attention
- LoRA (Low-Rank Adaptation) on Q,V projections
- Condition embedding for experimental conditions
- [CLS] token output as cell representation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation (LoRA).

    Implements: y = Wx + (BA)x where B ∈ R^{d×r}, A ∈ R^{r×d_in}
    Only A and B are trainable; W is frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen pretrained weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = F.linear(x, self.weight, self.bias)
        # LoRA forward
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out


class MultiHeadAttentionLoRA(nn.Module):
    """Multi-head attention with LoRA on Q and V projections.

    Uses flash attention via F.scaled_dot_product_attention.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q and V use LoRA; K uses standard linear
        self.q_proj = LoRALinear(d_model, d_model, lora_rank, lora_alpha, lora_dropout)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = LoRALinear(d_model, d_model, lora_rank, lora_alpha, lora_dropout)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Flash attention (numerically stable, faster, less memory)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Post-norm transformer block with LoRA attention (matching scGPT/PCP)."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttentionLoRA(
            d_model, n_heads, dropout, lora_rank, lora_alpha, lora_dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Post-norm: norm(x + sublayer(x)) — matches scGPT/PCP
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class GeneExpressionTokenizer(nn.Module):
    """Tokenize gene expression using rank-value encoding.

    Each gene is represented by two embeddings:
    - Gene identity embedding (universal scGPT vocabulary via gene_id_map)
    - Expression value embedding (binned expression level)
    """

    def __init__(
        self,
        n_genes: int = 2000,
        n_bins: int = 51,
        d_model: int = 512,
        gene_vocab_size: int = 60697,
    ):
        super().__init__()
        self.gene_embedding = nn.Embedding(gene_vocab_size + 2, d_model)  # +2 for [CLS], [PAD]
        self.expr_embedding = nn.Embedding(n_bins + 1, d_model)   # +1 for zero expression
        self.n_genes = n_genes
        self.gene_vocab_size = gene_vocab_size

        # Special tokens
        self.cls_token_id = gene_vocab_size
        self.pad_token_id = gene_vocab_size + 1

        # Gene ID map: position index → scGPT token ID
        self.register_buffer(
            "gene_id_map",
            torch.arange(n_genes, dtype=torch.long),  # Default: identity mapping
        )

    def set_gene_id_map(self, gene_names: List[str], scgpt_vocab: Dict[str, int]):
        """Build gene_id_map from gene names to scGPT token IDs."""
        gene_ids = []
        n_mapped = 0
        for g in gene_names:
            tid = (scgpt_vocab.get(g)
                   or scgpt_vocab.get(g.upper())
                   or scgpt_vocab.get(g.capitalize()))
            if tid is not None:
                gene_ids.append(tid)
                n_mapped += 1
            else:
                gene_ids.append(self.pad_token_id)
        self.gene_id_map = torch.tensor(gene_ids, dtype=torch.long)
        print(f"  Gene mapping: {n_mapped}/{len(gene_names)} genes mapped to scGPT vocab "
              f"({n_mapped/len(gene_names)*100:.1f}%)")

    def forward(
        self,
        expression: torch.Tensor,  # (B, n_genes) rank-encoded
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embeddings: (B, n_genes+1, d_model) with [CLS] prepended
            mask: (B, n_genes+1) attention mask
        """
        B, G = expression.shape

        # Gene identity tokens via universal mapping
        gene_ids = self.gene_id_map.unsqueeze(0).expand(B, -1)
        gene_emb = self.gene_embedding(gene_ids)

        # Expression value tokens
        expr_emb = self.expr_embedding(expression.long())

        # Combine gene identity + expression value
        token_emb = gene_emb + expr_emb

        # Prepend [CLS] token
        cls_emb = self.gene_embedding(
            torch.full((B, 1), self.cls_token_id, device=expression.device)
        )
        embeddings = torch.cat([cls_emb, token_emb], dim=1)

        # Attention mask (all tokens active)
        mask = torch.ones(B, G + 1, device=expression.device)

        return embeddings, mask


class PRISMEncoder(nn.Module):
    """Full PRISM encoder: tokenizer + transformer + projection head.

    Architecture aligned with scGPT/PCP for weight transfer:
    1. Gene expression tokenization (rank-value encoding, universal gene vocab)
    2. Condition embedding injection
    3. Post-norm transformer backbone with LoRA
    4. [CLS] token extraction
    5. Optional niche context concatenation
    6. Projection head for contrastive learning
    7. L2 normalization
    """

    def __init__(
        self,
        n_genes: int = 2000,
        n_bins: int = 51,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 512,
        d_output: int = 256,
        dropout: float = 0.1,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        n_conditions: int = 2,
        projection_dims: list = None,
        use_niche: bool = False,
        niche_dim: int = 64,
        use_gradient_checkpoint: bool = True,
        gene_vocab_size: int = 60697,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_output = d_output
        self.n_genes = n_genes

        # Tokenizer with universal gene vocabulary
        self.tokenizer = GeneExpressionTokenizer(n_genes, n_bins, d_model, gene_vocab_size)

        # Condition embedding (WT=0, En1-cKO=1)
        self.condition_embedding = nn.Embedding(n_conditions, d_model)

        # Positional encoding (learned, not sinusoidal, for gene sequences)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_genes + 2, d_model) * 0.02  # +2 for CLS and condition
        )

        # Post-norm transformer blocks with gradient checkpointing
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                lora_rank, lora_alpha, lora_dropout,
                use_checkpoint=use_gradient_checkpoint,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Projection head for contrastive learning (LayerNorm instead of BatchNorm for AMP stability)
        proj_dims = projection_dims or [512, 256, 128]
        proj_layers = []
        in_dim = d_model
        if use_niche:
            in_dim += niche_dim

        for out_dim in proj_dims[:-1]:
            proj_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            ])
            in_dim = out_dim
        proj_layers.append(nn.Linear(in_dim, proj_dims[-1]))
        self.projection_head = nn.Sequential(*proj_layers)

        # Reconstruction head (for masked gene prediction, prevents catastrophic forgetting)
        self.reconstruction_head = nn.Linear(d_model, n_genes)

        self.use_niche = use_niche

    def encode(
        self,
        expression: torch.Tensor,
        genotype: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode cells to [CLS] representation (before projection head).

        Args:
            expression: (B, n_genes) rank-encoded expression
            genotype: (B,) condition labels (0=WT, 1=En1-cKO)

        Returns:
            cls_repr: (B, d_model) [CLS] token representation
        """
        B = expression.shape[0]

        # Tokenize expression
        embeddings, attn_mask = self.tokenizer(expression)

        # Add condition embedding as second token
        cond_emb = self.condition_embedding(genotype).unsqueeze(1)
        embeddings = torch.cat([embeddings[:, :1], cond_emb, embeddings[:, 1:]], dim=1)

        # Add positional encoding
        seq_len = embeddings.shape[1]
        embeddings = embeddings + self.pos_embedding[:, :seq_len]

        # Transformer forward
        for block in self.transformer_blocks:
            embeddings = block(embeddings)

        embeddings = self.norm(embeddings)

        # Extract [CLS] token (first position)
        cls_repr = embeddings[:, 0]

        return cls_repr

    def forward(
        self,
        expression: torch.Tensor,
        genotype: torch.Tensor,
        niche_context: Optional[torch.Tensor] = None,
        return_reconstruction: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode + project.

        Returns tuple (z, cls_repr, recon) so DataParallel can gather
        across GPUs. recon is zeros if return_reconstruction is False.

        z: L2-normalized projected embedding (B, d_output)
        cls_repr: [CLS] representation before projection (B, d_model)
        recon: reconstructed expression (B, n_genes) or zeros
        """
        cls_repr = self.encode(expression, genotype)

        # Optional niche context concatenation
        proj_input = cls_repr
        if self.use_niche and niche_context is not None:
            proj_input = torch.cat([cls_repr, niche_context], dim=-1)

        # Project and L2 normalize
        z = self.projection_head(proj_input)
        z = F.normalize(z, dim=-1)

        if return_reconstruction:
            recon = self.reconstruction_head(cls_repr)
        else:
            recon = torch.zeros(
                expression.shape[0], self.n_genes,
                device=expression.device, dtype=z.dtype,
            )

        return z, cls_repr, recon

    def get_trainable_params(self):
        """Return only LoRA + projection head parameters (for optimizer)."""
        trainable = []
        for name, param in self.named_parameters():
            if "lora_" in name or "projection" in name or "condition" in name:
                param.requires_grad = True
                trainable.append(param)
            else:
                param.requires_grad = False
        return trainable

    def get_all_trainable_params(self):
        """Return separate param groups for different learning rates."""
        lora_params = []
        head_params = []
        other_params = []

        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                lora_params.append(param)
            elif "projection" in name or "reconstruction" in name:
                param.requires_grad = True
                head_params.append(param)
            elif "condition" in name or "pos_embedding" in name:
                param.requires_grad = True
                other_params.append(param)
            else:
                param.requires_grad = False

        return [
            {"params": lora_params, "lr_group": "lora"},
            {"params": head_params, "lr_group": "head"},
            {"params": other_params, "lr_group": "lora"},
        ]

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
