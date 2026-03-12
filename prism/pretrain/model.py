"""PCP Encoder: Transformer for perturbation-contrastive pre-training.

Architecture aligned with scGPT for weight transfer:
- Universal gene vocabulary (60,697 scGPT tokens + CLS + PAD)
- Post-norm transformer blocks (matching scGPT)
- Flash attention via scaled_dot_product_attention
- scGPT weight loading with fused QKV splitting
- Gene ID mapping for universal vocabulary transfer
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention (no LoRA)."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Use flash attention (numerically stable in fp16, faster, less memory)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Post-norm transformer block (matching scGPT architecture)."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
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
        # Post-norm: norm(x + sublayer(x))
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class PCPEncoder(nn.Module):
    """Perturbation-Contrastive Pre-training Encoder.

    Architecture aligned with scGPT for weight transfer:
    1. Universal gene vocabulary (60,697 scGPT tokens + CLS + PAD)
    2. Gene ID mapping via registered buffer (position → scGPT token ID)
    3. Post-norm transformer backbone (no LoRA — all weights trainable)
    4. [CLS] token extraction
    5. Contrastive projection head (L2-normalized)
    6. MLM head (predict masked gene expression bins)

    Weight transfer chain:
        scGPT → PCPEncoder (Stage B pre-training) → PRISMEncoder (downstream LoRA fine-tuning)
    """

    def __init__(
        self,
        n_genes: int = 2000,
        n_bins: int = 51,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 512,
        d_output: int = 128,
        dropout: float = 0.1,
        projection_dims: list = None,
        use_gradient_checkpoint: bool = True,
        scgpt_vocab_size: int = 60697,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.n_bins = n_bins
        self.d_model = d_model
        self.mask_token_id = n_bins  # Use n_bins as MASK token (valid bins: 0..n_bins-1)
        self.scgpt_vocab_size = scgpt_vocab_size

        # Gene identity embedding: universal scGPT vocabulary + CLS + PAD
        self.gene_embedding = nn.Embedding(scgpt_vocab_size + 2, d_model)
        self.cls_token_id = scgpt_vocab_size
        self.pad_token_id = scgpt_vocab_size + 1

        # Expression value embedding (+2 for zero expression AND mask token)
        self.expr_embedding = nn.Embedding(n_bins + 2, d_model)

        # Gene ID map: position index → scGPT token ID (set via set_gene_id_map)
        self.register_buffer(
            "gene_id_map",
            torch.arange(n_genes, dtype=torch.long),  # Default: identity mapping
        )

        # Positional encoding (learned, position-specific, not gene-specific)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_genes + 1, d_model) * 0.02  # +1 for CLS
        )

        # Transformer backbone (post-norm, matching scGPT)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                use_checkpoint=use_gradient_checkpoint,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Contrastive projection head
        proj_dims = projection_dims or [512, 256, 128]
        proj_layers = []
        in_dim = d_model
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

        # MLM head: predict expression bin for masked gene positions
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_bins + 1),  # Predict bins 0..n_bins (51 classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following BERT-style initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_gene_id_map(self, gene_names: List[str], scgpt_vocab: Dict[str, int]):
        """Build gene_id_map from gene names to scGPT token IDs.

        For each gene, tries exact match, then uppercase (mouse→human),
        then capitalized. Falls back to pad_token_id if not found.

        Args:
            gene_names: List of gene names (corpus or downstream HVGs)
            scgpt_vocab: Dict mapping gene name → scGPT token ID
        """
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

    def tokenize(self, expression: torch.Tensor) -> torch.Tensor:
        """Convert rank-encoded expression to embeddings with [CLS] prepended.

        Uses gene_id_map to look up universal scGPT gene embeddings.

        Args:
            expression: (B, n_genes) rank-encoded, may contain mask_token_id

        Returns:
            embeddings: (B, n_genes+1, d_model)
        """
        B, G = expression.shape

        # Gene identity tokens via universal mapping
        gene_ids = self.gene_id_map.unsqueeze(0).expand(B, -1)  # (B, n_genes)
        gene_emb = self.gene_embedding(gene_ids)

        # Expression value tokens (including MASK token at index n_bins)
        expr_emb = self.expr_embedding(expression.long())

        # Combine
        token_emb = gene_emb + expr_emb

        # Prepend [CLS]
        cls_emb = self.gene_embedding(
            torch.full((B, 1), self.cls_token_id, device=expression.device)
        )
        embeddings = torch.cat([cls_emb, token_emb], dim=1)

        return embeddings

    def encode(self, expression: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode expression to transformer output.

        Returns:
            cls_repr: (B, d_model) [CLS] representation
            all_hidden: (B, n_genes+1, d_model) all position outputs
        """
        embeddings = self.tokenize(expression)

        # Add positional encoding
        seq_len = embeddings.shape[1]
        embeddings = embeddings + self.pos_embedding[:, :seq_len]

        # Transformer forward
        for block in self.transformer_blocks:
            embeddings = block(embeddings)

        embeddings = self.norm(embeddings)

        cls_repr = embeddings[:, 0]  # [CLS] token
        return cls_repr, embeddings

    def forward(
        self,
        expression: torch.Tensor,
        masked_expression: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass for pre-training.

        For DataParallel compatibility, returns fixed-size outputs:
        - MLM logits are computed for ALL gene positions (not just masked ones)
        - Masking for loss is applied in the trainer

        Args:
            expression: (B, n_genes) original rank-encoded expression (targets)
            masked_expression: (B, n_genes) expression with MASK tokens at masked positions
            mask: (B, n_genes) boolean mask (True = masked positions)

        Returns:
            z: (B, d_output) L2-normalized contrastive embeddings
            mlm_logits: (B, n_genes, n_bins+1) predictions for all gene positions
            cls_repr: (B, d_model) [CLS] representation
        """
        # Encode masked input
        cls_repr, all_hidden = self.encode(masked_expression)

        # Contrastive: project [CLS] and L2-normalize
        z = self.projection_head(cls_repr)
        z = F.normalize(z, dim=-1)

        # MLM: predict expression bins for ALL positions (filter in loss)
        gene_hidden = all_hidden[:, 1:]  # (B, n_genes, d_model)
        mlm_logits = self.mlm_head(gene_hidden)  # (B, n_genes, n_bins+1)

        return z, mlm_logits, cls_repr

    @torch.no_grad()
    def get_embeddings(self, expression: torch.Tensor) -> torch.Tensor:
        """Extract contrastive embeddings (no masking, for inference)."""
        cls_repr, _ = self.encode(expression)
        z = self.projection_head(cls_repr)
        z = F.normalize(z, dim=-1)
        return z

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @staticmethod
    def load_scgpt_weights(
        encoder: "PCPEncoder",
        scgpt_checkpoint_path: str,
        scgpt_vocab_path: str = None,
        corpus_gene_names: List[str] = None,
    ) -> Dict[str, str]:
        """Load pre-trained scGPT weights into PCPEncoder.

        Handles:
        1. Gene embedding transfer (60697 → 60699 with CLS/PAD)
        2. Fused Wqkv splitting into separate Q/K/V
        3. FFN weight mapping (linear1→ff.0, linear2→ff.3)
        4. Layer norm copying
        5. Expression embedding pre-computation from scGPT value_encoder
        6. Gene ID mapping from corpus genes to scGPT vocab

        Args:
            encoder: PCPEncoder instance to load weights into
            scgpt_checkpoint_path: Path to scGPT best_model.pt
            scgpt_vocab_path: Path to scGPT vocab.json (for gene ID mapping)
            corpus_gene_names: List of gene names in the corpus vocabulary

        Returns:
            Transfer log dict mapping parameter names to transfer status
        """
        transfer_log = {}

        # Load scGPT checkpoint
        scgpt_sd = torch.load(scgpt_checkpoint_path, map_location="cpu", weights_only=False)
        print(f"  Loaded scGPT checkpoint: {len(scgpt_sd)} parameters")

        # 1. Gene embeddings: scGPT [60697, 512] → encoder [60699, 512]
        scgpt_gene_emb = scgpt_sd["encoder.embedding.weight"]
        with torch.no_grad():
            encoder.gene_embedding.weight[:scgpt_gene_emb.shape[0]] = scgpt_gene_emb
        transfer_log["encoder.embedding.weight"] = (
            f"transferred [{scgpt_gene_emb.shape}] → gene_embedding[:60697]"
        )

        # 2. Expression embeddings: pre-compute from scGPT value_encoder MLP
        # value_encoder: scalar → Linear(1,512) → ReLU → Linear(512,512) → LayerNorm
        ve_w1 = scgpt_sd["value_encoder.linear1.weight"]  # [512, 1]
        ve_b1 = scgpt_sd["value_encoder.linear1.bias"]    # [512]
        ve_w2 = scgpt_sd["value_encoder.linear2.weight"]  # [512, 512]
        ve_b2 = scgpt_sd["value_encoder.linear2.bias"]    # [512]
        ve_nw = scgpt_sd["value_encoder.norm.weight"]      # [512]
        ve_nb = scgpt_sd["value_encoder.norm.bias"]        # [512]

        with torch.no_grad():
            # Compute embeddings for expression bins 0-50, normalized to [0,1]
            bins = torch.arange(0, encoder.n_bins).float().unsqueeze(1) / float(encoder.n_bins - 1)  # (51, 1)
            # Forward through value_encoder: linear1 → ReLU → linear2 → LayerNorm
            h = F.relu(F.linear(bins, ve_w1, ve_b1))  # (51, 512)
            h = F.linear(h, ve_w2, ve_b2)              # (51, 512)
            # LayerNorm
            h = F.layer_norm(h, [h.shape[-1]], ve_nw, ve_nb)  # (51, 512)
            encoder.expr_embedding.weight[:encoder.n_bins] = h
        transfer_log["value_encoder"] = f"pre-computed {encoder.n_bins} expression embeddings from MLP"

        # 3. Transformer layers: split fused Wqkv, copy FFN and norms
        n_layers = len(encoder.transformer_blocks)
        for i in range(n_layers):
            prefix = f"transformer_encoder.layers.{i}"

            # Check if layer exists in scGPT
            qkv_key = f"{prefix}.self_attn.Wqkv.weight"
            if qkv_key not in scgpt_sd:
                transfer_log[f"layer_{i}"] = "skipped (not in scGPT checkpoint)"
                continue

            block = encoder.transformer_blocks[i]

            # Split fused QKV [1536, 512] → Q [512,512], K [512,512], V [512,512]
            Wqkv = scgpt_sd[f"{prefix}.self_attn.Wqkv.weight"]
            bqkv = scgpt_sd[f"{prefix}.self_attn.Wqkv.bias"]
            q_w, k_w, v_w = Wqkv.chunk(3, dim=0)
            q_b, k_b, v_b = bqkv.chunk(3, dim=0)

            with torch.no_grad():
                block.attn.q_proj.weight.copy_(q_w)
                block.attn.q_proj.bias.copy_(q_b)
                block.attn.k_proj.weight.copy_(k_w)
                block.attn.k_proj.bias.copy_(k_b)
                block.attn.v_proj.weight.copy_(v_w)
                block.attn.v_proj.bias.copy_(v_b)
            transfer_log[qkv_key] = f"split [1536,512] → Q/K/V [{q_w.shape}] each"

            # Output projection
            o_w = scgpt_sd[f"{prefix}.self_attn.out_proj.weight"]
            o_b = scgpt_sd[f"{prefix}.self_attn.out_proj.bias"]
            with torch.no_grad():
                block.attn.o_proj.weight.copy_(o_w)
                block.attn.o_proj.bias.copy_(o_b)
            transfer_log[f"{prefix}.self_attn.out_proj"] = "transferred"

            # FFN: scGPT linear1 → ff.0, linear2 → ff.3
            with torch.no_grad():
                block.ff[0].weight.copy_(scgpt_sd[f"{prefix}.linear1.weight"])
                block.ff[0].bias.copy_(scgpt_sd[f"{prefix}.linear1.bias"])
                block.ff[3].weight.copy_(scgpt_sd[f"{prefix}.linear2.weight"])
                block.ff[3].bias.copy_(scgpt_sd[f"{prefix}.linear2.bias"])
            transfer_log[f"{prefix}.linear1/2"] = "transferred → ff.0/ff.3"

            # Layer norms
            with torch.no_grad():
                block.norm1.weight.copy_(scgpt_sd[f"{prefix}.norm1.weight"])
                block.norm1.bias.copy_(scgpt_sd[f"{prefix}.norm1.bias"])
                block.norm2.weight.copy_(scgpt_sd[f"{prefix}.norm2.weight"])
                block.norm2.bias.copy_(scgpt_sd[f"{prefix}.norm2.bias"])
            transfer_log[f"{prefix}.norm1/2"] = "transferred"

        # 4. Final layer norm: scGPT encoder.enc_norm → encoder.norm
        with torch.no_grad():
            encoder.norm.weight.copy_(scgpt_sd["encoder.enc_norm.weight"])
            encoder.norm.bias.copy_(scgpt_sd["encoder.enc_norm.bias"])
        transfer_log["encoder.enc_norm"] = "transferred → norm"

        # 5. Gene ID mapping (if vocab provided)
        if scgpt_vocab_path and corpus_gene_names:
            with open(scgpt_vocab_path) as f:
                scgpt_vocab = json.load(f)
            encoder.set_gene_id_map(corpus_gene_names, scgpt_vocab)
            transfer_log["gene_id_map"] = f"mapped {len(corpus_gene_names)} corpus genes"

        # Summary
        n_transferred = sum(1 for v in transfer_log.values() if "transferred" in v or "split" in v or "pre-computed" in v or "mapped" in v)
        print(f"  scGPT weight transfer: {n_transferred} operations completed")
        for key, status in transfer_log.items():
            print(f"    {key}: {status}")

        return transfer_log

    def transfer_weights_to_prism(self, prism_encoder) -> Dict[str, str]:
        """Transfer pre-trained weights to a PRISMEncoder for fine-tuning.

        Maps standard linear Q/V projections to LoRALinear frozen weights.
        Handles universal gene vocabulary (60699 gene embeddings + gene_id_map).
        Handles partial transfers for pos_embedding (PCP lacks condition token)
        and expr_embedding (PCP has MASK token, PRISM doesn't).

        Returns:
            Dict mapping of transferred parameter names and their status.
        """
        transfer_log = {}
        src = self.state_dict()
        tgt = prism_encoder.state_dict()

        for src_key, src_param in src.items():
            # Map PCP parameter names to PRISM parameter names
            tgt_key = self._map_param_name(src_key)
            if tgt_key is None:
                transfer_log[src_key] = "skipped (no mapping)"
                continue

            if tgt_key in tgt:
                if src_param.shape == tgt[tgt_key].shape:
                    tgt[tgt_key] = src_param
                    transfer_log[src_key] = f"transferred -> {tgt_key}"
                elif src_key == "pos_embedding":
                    # PCP: (1, n_genes+1, d_model) — CLS + genes
                    # PRISM: (1, n_genes+2, d_model) — CLS + condition + genes
                    # Copy CLS pos to CLS, gene positions shifted by 1 for condition token
                    min_len = min(src_param.shape[1], tgt[tgt_key].shape[1])
                    tgt[tgt_key][:, :1] = src_param[:, :1]  # CLS position
                    tgt[tgt_key][:, 2:min_len+1] = src_param[:, 1:min_len]  # Gene positions (offset by condition)
                    transfer_log[src_key] = f"partial transferred (CLS + {min_len-1} gene positions)"
                elif src_key == "expr_embedding.weight":
                    # PCP: (n_bins+2, d_model) — bins + MASK token
                    # PRISM: (n_bins+1, d_model) — bins only (no MASK)
                    min_bins = min(src_param.shape[0], tgt[tgt_key].shape[0])
                    tgt[tgt_key][:min_bins] = src_param[:min_bins]
                    transfer_log[src_key] = f"partial transferred ({min_bins} bins)"
                else:
                    transfer_log[src_key] = f"shape mismatch: {src_param.shape} vs {tgt[tgt_key].shape}"
            else:
                transfer_log[src_key] = f"target key {tgt_key} not found"

        prism_encoder.load_state_dict(tgt, strict=False)
        n_transferred = sum(1 for v in transfer_log.values() if "transferred" in v or "partial" in v)
        print(f"Transferred {n_transferred}/{len(src)} parameters")
        return transfer_log

    def _map_param_name(self, src_key: str) -> Optional[str]:
        """Map PCP encoder parameter name to PRISMEncoder name."""
        # gene_embedding -> tokenizer.gene_embedding
        if src_key.startswith("gene_embedding."):
            return src_key.replace("gene_embedding.", "tokenizer.gene_embedding.")
        if src_key.startswith("expr_embedding."):
            return src_key.replace("expr_embedding.", "tokenizer.expr_embedding.")

        # gene_id_map buffer -> tokenizer.gene_id_map
        if src_key == "gene_id_map":
            return "tokenizer.gene_id_map"

        # transformer_blocks.N.attn.{q,v}_proj -> same but LoRALinear weight
        if ".attn.q_proj." in src_key or ".attn.v_proj." in src_key:
            return src_key  # Same key names, different class

        # Direct mappings (same key names)
        if any(k in src_key for k in [
            "transformer_blocks.", "pos_embedding", "norm.",
            "projection_head.", ".k_proj.", ".o_proj.",
        ]):
            return src_key

        # MLM head doesn't transfer (PRISM has reconstruction_head instead)
        if "mlm_head" in src_key:
            return None

        return src_key
