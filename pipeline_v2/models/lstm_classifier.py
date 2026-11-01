"""
pipeline_v2/models/lstm_classifier.py

Model 1 — Bi-LSTM Sign Language Classifier
===========================================
Architecture:
    Input (B, T=32, F=378)
    → Spatial Embedding: Linear → BN → ReLU → Dropout  (×2)
    → Temporal Conv1D:   Conv1d(H, H, k=3, pad=1) → BN → ReLU → Dropout
    → Bi-LSTM:           2 layers, hidden=H, bidirectional
    → Classification:    Linear(2H→H) → ReLU → Dropout → Linear(H→N)

This replaces the BiGRU baseline from our previous experiments.
LSTM adds an explicit cell-state gating mechanism that can maintain longer-term
context — useful for sentences with many sequential hand shapes.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self,
                 input_size:  int = 378,
                 hidden_size: int = 256,
                 num_classes: int = 97,
                 num_layers:  int = 2,
                 dropout:     float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size

        # ── 1. Spatial Embedding ──────────────────────────────────────────────
        # Applied independently per frame: (B*T, F) → (B*T, H)
        self.spatial_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── 2. Temporal Convolution ───────────────────────────────────────────
        # Captures short-range temporal dynamics (wrist flicks, finger transitions)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── 3. Bi-LSTM ────────────────────────────────────────────────────────
        # LSTM cell state gives an explicit "memory slot" vs GRU's merged state.
        # dropout between LSTM layers (only active when num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 4. Classification Head ────────────────────────────────────────────
        # Bi-LSTM concatenates forward + backward final states → 2H features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, F)
        Returns:
            logits : (B, num_classes)
        """
        B, T, F = x.shape

        # Spatial embedding (frame-wise)
        x = x.reshape(B * T, F)
        x = self.spatial_embedding(x)   # (B*T, H)

        # Temporal convolution
        x = x.view(B, T, -1).transpose(1, 2)   # (B, H, T)
        x = self.temporal_conv(x)               # (B, H, T)
        x = x.transpose(1, 2)                   # (B, T, H)

        # Bi-LSTM
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * 2, B, H) — take last layer's fwd + bwd
        h_fwd = h_n[-2, :, :]   # (B, H)
        h_bwd = h_n[-1, :, :]   # (B, H)
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)

        return self.classifier(h_cat)


if __name__ == "__main__":
    # Quick sanity check
    model  = LSTMClassifier(input_size=378, hidden_size=256, num_classes=97)
    x_test = torch.randn(4, 32, 378)
    out    = model(x_test)
    print(f"LSTMClassifier output shape: {out.shape}")   # expect (4, 97)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

