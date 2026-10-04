import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Face landmark indices corresponding to lips and eyebrows in MediaPipe Face Mesh.
# We offset them by 75 because Face landmarks start at index 75 in our 543-node array.
FACE_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 
             178, 181, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 
             321, 324, 375, 402, 405, 409, 415]
FACE_EYEBROWS = [46, 52, 53, 55, 65, 70, 276, 282, 283, 285, 295, 300]
FACE_SUBSET = sorted(FACE_LIPS + FACE_EYEBROWS)
FACE_INDICES = [75 + idx for idx in FACE_SUBSET]

# Pose (0-32), LHand (33-53), RHand (54-74)
POSE_INDICES = list(range(0, 33))
LHAND_INDICES = list(range(33, 54))
RHAND_INDICES = list(range(54, 75))

# Final subset to use: 33 + 21 + 21 + 51 = 126 landmarks
SELECTED_INDICES = POSE_INDICES + LHAND_INDICES + RHAND_INDICES + FACE_INDICES

class ISLDataset(Dataset):
    def __init__(self, h5_path: str, augment: bool = False):
        """
        PyTorch Dataset for ISL-CSLRT HDF5 data.
        
        Args:
            h5_path: Path to the .h5 file.
            augment: If True, apply spatial jitter (noise) for training.
        """
        super().__init__()
        self.h5_path = h5_path
        self.augment = augment
        
        # We load all data into memory since it's very small (<100MB per file)
        # This makes training extremely fast.
        with h5py.File(self.h5_path, "r") as hf:
            self.clips = hf["clips"][:]       # (N, 32, 543, 3)
            self.labels = hf["labels"][:]     # (N,)
            
        # Select the specific landmarks
        self.clips = self.clips[:, :, SELECTED_INDICES, :]  # (N, 32, 126, 3)
        
        self.n_samples = self.clips.shape[0]
        self.n_frames = self.clips.shape[1]
        self.n_landmarks = self.clips.shape[2]
        self.n_coords = self.clips.shape[3]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Shape: (32, 126, 3)
        x = self.clips[idx].copy()
        y = self.labels[idx]
        
        if self.augment:
            # Spatial Augmentation: add slight gaussian noise to coordinates.
            # standard deviation of 0.01 (1% of shoulder width).
            noise = np.random.normal(loc=0.0, scale=0.01, size=x.shape).astype(np.float32)
            x += noise
            
        # Flatten the spatial dimension into a single feature vector per frame
        # Shape becomes: (32, 126 * 3) -> (32, 378)
        x = x.reshape(self.n_frames, -1)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_dataloaders(train_h5, val_h5, batch_size=32):
    """Utility to instantly create train and val dataloaders."""
    train_dataset = ISLDataset(train_h5, augment=True)
    val_dataset = ISLDataset(val_h5, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader
