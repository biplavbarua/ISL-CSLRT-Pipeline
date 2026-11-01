"""
pipeline_v2/models/pose_transformer.py

Model 3a — Pose Transformer Encoder
=====================================
A Transformer encoder that processes the T=32 keypoint sequence and produces
a sentence-level embedding. Used standalone as a classifier OR as the visual
encoder feeding into the Gemma 2B LLM decoder (llm_decoder.py).

Architecture:
    Input: (B, T, 378)
    → Linear projection → (B, T, d_model)
    → + Learnable positional encoding
    → Transformer encoder (n_heads=8, n_layers=4, d_ff=1024)
    → CLS token pooling → (B, d_model)
    → Classification head → (B, num_classes)
"""

import torch
import torch.nn as nn
import math


class PoseTransformer(nn.Module):
    def __init__(self,
                 input_size:  int   = 378,
                 d_model:     int   = 256,
                 n_heads:     int   = 8,
                 n_layers:    int   = 4,
                 d_ff:        int   = 1024,
                 num_classes: int   = 97,
                 max_seq_len: int   = 32,
                 dropout:     float = 0.1):
        super().__init__()
        self.d_model = d_model

        # ── 1. Input projection ───────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        # ── 2. CLS token (prepended; pooled for classification) ───────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 3. Learnable Positional Encoding ─────────────────────────────────
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── 4. Transformer Encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN Transformer (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── 5. Classification head ────────────────────────────────────────────
        self.head_dropout = nn.Dropout(dropout)
        self.classifier   = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the CLS embedding without classification head.
        Used as the visual encoder output for the LLM decoder.

        Args:
            x : (B, T, F)
        Returns:
            embedding : (B, d_model)
        """
        B = x.shape[0]
        x = self.input_proj(x)   # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)           # (B, T+1, d_model)

        # Add positional encoding (truncate if sequence is shorter than max)
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer
        x = self.transformer(x)   # (B, T+1, d_model)

        # Pool CLS token
        return x[:, 0, :]   # (B, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, F)
        Returns:
            logits : (B, num_classes)
        """
        emb = self.encode(x)
        return self.classifier(self.head_dropout(emb))


if __name__ == "__main__":
    model  = PoseTransformer(input_size=378, d_model=256, num_classes=97)
    x_test = torch.randn(4, 32, 378)
    out    = model(x_test)
    print(f"PoseTransformer output shape: {out.shape}")   # expect (4, 97)
    emb    = model.encode(x_test)
    print(f"PoseTransformer embedding shape: {emb.shape}")  # expect (4, 256)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

