"""
pipeline_v2/models/stgcn.py

Model 2 — Spatial-Temporal Graph Convolutional Network (ST-GCN)
===============================================================
Reference: Yan et al., "Spatial Temporal Graph Convolutional Networks
           for Skeleton-Based Action Recognition", AAAI 2018.

We adapt ST-GCN for the 75 pose+hand landmarks of ISL-CSLRT.

Graph nodes  : 75 landmarks (33 Pose + 21 L-Hand + 21 R-Hand)
              (face dropped from graph — no topological structure to exploit)
Graph edges  : Body skeleton + hand bone edges (defined below)
Input shape  : (B, 3, T, 75)  — channels=3 (x,y,z), T=32 frames, V=75 nodes
Output       : (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── Skeleton Adjacency Definition ─────────────────────────────────────────────
# Pose landmark indices 0–32 (MediaPipe convention)
POSE_EDGES = [
    (0,1),(1,2),(2,3),(3,7),        # face outline
    (0,4),(4,5),(5,6),(6,8),
    (9,10),                          # mouth
    (11,12),                         # shoulders
    (11,13),(13,15),(15,17),(17,19),(19,15),(15,21),  # left arm
    (12,14),(14,16),(16,18),(18,20),(20,16),(16,22),  # right arm
    (11,23),(12,24),(23,24),         # hips
    (23,25),(25,27),(27,29),(29,31),(31,27),  # left leg
    (24,26),(26,28),(28,30),(30,32),(32,28),  # right leg
]

# Left hand (nodes 33–53): wrist + 4 fingers × 4 joints + thumb × 4 joints
LHAND_WRIST = 33
LHAND_EDGES = []
finger_starts = [34, 38, 42, 46, 50]   # index, middle, ring, pinky, thumb
for fs in finger_starts:
    LHAND_EDGES.append((LHAND_WRIST, fs))     # wrist → knuckle
    for j in range(3):
        LHAND_EDGES.append((fs + j, fs + j + 1))

# Right hand (nodes 54–74)
RHAND_WRIST = 54
RHAND_EDGES = [(e[0] + 21, e[1] + 21) for e in LHAND_EDGES]

# Cross-body edges: shoulder ↔ wrist anchor
CROSS_EDGES = [(11, LHAND_WRIST), (12, RHAND_WRIST)]

ALL_EDGES = POSE_EDGES + LHAND_EDGES + RHAND_EDGES + CROSS_EDGES
N_NODES   = 75  # pose + lhand + rhand (no face landmarks in graph)


def build_adjacency(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Build symmetric normalised adjacency matrix."""
    A = np.eye(n, dtype=np.float32)
    for (i, j) in edges:
        if i < n and j < n:
            A[i, j] = 1.0
            A[j, i] = 1.0
    # Row-normalise
    D = np.diag(A.sum(axis=1) ** -0.5)
    return D @ A @ D


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial graph conv + temporal conv."""

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 t_kernel: int = 9, stride: int = 1, dropout: float = 0.5):
        super().__init__()
        self.register_buffer("A", A)

        # Spatial GCN: (B, C, T, V) × (V, V) → (B, C', T, V)
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        # Temporal conv: across time dimension T
        pad = (t_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(t_kernel, 1),
                      padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )

        # Residual connection
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        res = self.residual(x)

        # Spatial graph conv
        x = torch.einsum("bctv,vw->bctw", x, self.A)   # aggregate neighbours
        x = self.gcn(x)

        # Temporal conv + residual
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNClassifier(nn.Module):
    def __init__(self,
                 in_channels:  int   = 3,
                 num_classes:  int   = 97,
                 n_nodes:      int   = N_NODES,
                 dropout:      float = 0.5):
        super().__init__()

        A_np = build_adjacency(n_nodes, ALL_EDGES)
        A    = torch.from_numpy(A_np)

        # Input BN
        self.data_bn = nn.BatchNorm1d(in_channels * n_nodes)

        # ST-GCN stack: gradually double channels
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64,  A, dropout=dropout),
            STGCNBlock(64,          64,  A, dropout=dropout),
            STGCNBlock(64,          64,  A, dropout=dropout),
            STGCNBlock(64,          128, A, dropout=dropout, stride=2),
            STGCNBlock(128,         128, A, dropout=dropout),
            STGCNBlock(128,         128, A, dropout=dropout),
            STGCNBlock(128,         256, A, dropout=dropout, stride=2),
            STGCNBlock(256,         256, A, dropout=dropout),
            STGCNBlock(256,         256, A, dropout=dropout),
        ])

        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, T, V) where V=75 (pose+hands only)
        Returns:
            logits : (B, num_classes)
        """
        B, C, T, V = x.shape

        # Input BatchNorm (flatten V×C → treat as 1D features)
        x = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)
        x = self.data_bn(x)
        x = x.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()  # (B, C, T, V)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x).squeeze(-1).squeeze(-1)   # (B, 256)
        return self.classifier(x)


if __name__ == "__main__":
    model  = STGCNClassifier(in_channels=3, num_classes=97)
    x_test = torch.randn(4, 3, 32, 75)
    out    = model(x_test)
    print(f"STGCNClassifier output shape: {out.shape}")   # expect (4, 97)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

