"""
pipeline_v2/models/dataset.py

Shared PyTorch Dataset for all three models.
Loads keypoints from HDF5 files produced by step05_export.py.

Each item: (clip_tensor, label_tensor)
    clip_tensor : (T, 378)  float32  — flattened 126 landmarks × 3 coords
    label_tensor: ()        long
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py


class ISLDataset(Dataset):
    def __init__(self, h5_path: str, augment: bool = False, flatten: bool = True):
        """
        Parameters
        ----------
        h5_path : path to HDF5 file from step05
        augment : if True, applies spatial jitter + mirror-flip augmentation
        flatten : if True, output shape is (T, 126*3=378); else (T, 126, 3)
        """
        with h5py.File(h5_path, "r") as hf:
            # clips shape: (N, T, 126, 3)
            self.clips      = hf["clips"][:]
            self.labels     = hf["labels"][:]
            self.signer_ids = hf["signer_ids"][:]
            self.label_map  = json.loads(hf.attrs["label_map"])

        self.augment = augment
        self.flatten = flatten
        self.T       = self.clips.shape[1]
        self.n_lm    = self.clips.shape[2]  # 126
        self.n_coord = self.clips.shape[3]  # 3

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        x = self.clips[idx].copy()   # (T, 126, 3)

        if self.augment:
            # 1. Spatial jitter — small Gaussian noise (1% of shoulder width)
            x += np.random.normal(0.0, 0.01, x.shape).astype(np.float32)

            # 2. Mirror-flip (ISL is right-hand dominant; mirroring creates
            #    a plausible "left-handed" signer variant)
            if np.random.rand() < 0.5:
                x[:, :, 0] = -x[:, :, 0]   # negate x-axis

            # 3. Temporal frame dropout (replace 1–3 random frames with zeros)
            if np.random.rand() < 0.4:
                n_drop = np.random.randint(1, 4)
                start  = np.random.randint(0, self.T - n_drop)
                x[start:start + n_drop] = 0.0

        x_tensor = torch.from_numpy(x)   # (T, 126, 3)

        if self.flatten:
            # (T, 126, 3) → (T, 378)
            x_tensor = x_tensor.view(self.T, -1)

        return x_tensor, torch.tensor(self.labels[idx], dtype=torch.long)


def get_dataloaders(train_h5: str, val_h5: str, batch_size: int = 32,
                    augment: bool = True, flatten: bool = True):
    train_ds = ISLDataset(train_h5, augment=augment, flatten=flatten)
    val_ds   = ISLDataset(val_h5,   augment=False,   flatten=flatten)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)
    return train_dl, val_dl

