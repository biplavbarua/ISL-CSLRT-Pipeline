import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, input_size=378, hidden_size=256, num_classes=97, dropout=0.5):
        """
        Spatial-Temporal 1D-CNN + Bi-GRU network for Sign Language Recognition.
        
        Args:
            input_size: Number of features per frame (e.g. 126 landmarks * 3 = 378).
            hidden_size: Hidden dimension for embeddings and GRU.
            num_classes: Number of sentence classes to predict.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        
        # 1. Spatial Embedding
        # Flattens the 3D coordinates into a rich feature vector per frame.
        self.spatial_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Temporal Convolution (1D-CNN)
        # Captures local temporal dynamics (e.g. quick wrist flicks across 3 frames).
        # Padding=1 keeps the sequence length the same (32 frames).
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Sequence Modeling (Bi-GRU)
        # Captures long-term context forwards and backwards.
        self.gru = nn.GRU(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout
        )
        
        # 4. Classification Head
        # Bi-GRU outputs hidden_size * 2 features (forward + backward).
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Time, Features) -> (B, 32, 378)
        Returns:
            logits: Tensor of shape (Batch, NumClasses) -> (B, 96)
        """
        B, T, F = x.shape
        
        # -- Spatial Embedding --
        # Reshape to (B*T, F) to apply Linear layers to all frames independently
        x = x.view(B * T, F)
        x = self.spatial_embedding(x)
        
        # -- Temporal Convolution --
        # Reshape back to (B, T, Hidden) -> transpose to (B, Hidden, T) for Conv1d
        x = x.view(B, T, -1).transpose(1, 2)
        x = self.temporal_conv(x)
        
        # -- Sequence Modeling --
        # Transpose back to (B, T, Hidden) for GRU
        x = x.transpose(1, 2)
        
        # Pass through GRU. We ignore the output sequence (out) and just use the 
        # final hidden state (h_n) to classify the whole sentence.
        out, h_n = self.gru(x)
        
        # h_n shape: (num_layers * num_directions, Batch, Hidden)
        # We want the final forward and backward states of the last layer:
        # h_n[-2, :, :] is forward, h_n[-1, :, :] is backward
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        
        # Concatenate forward and backward states -> (B, Hidden*2)
        h_final = torch.cat([h_forward, h_backward], dim=1)
        
        # -- Classification --
        logits = self.classifier(h_final)
        
        return logits
