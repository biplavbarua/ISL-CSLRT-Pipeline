import os
import h5py
import json
import time
import subprocess
import numpy as np
import torch
import pandas as pd
import sys

# 1. Restore baseline files
subprocess.run(["git", "restore", "training/train.py", "training/dataset.py"], check=True)

# 2. Pool all clips
src_files = [
    "preprocessed_v4/exports/standard_train.h5",
    "preprocessed_v4/exports/standard_val.h5",
    "preprocessed_v4/exports/standard_test.h5"
]

all_clips = []
all_labels = []
label_map = None

for f_path in src_files:
    with h5py.File(f_path, "r") as f:
        all_clips.append(f["clips"][:])
        all_labels.append(f["labels"][:])
        if label_map is None:
            label_map = f.attrs["label_map"]

all_clips = np.concatenate(all_clips, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

num_clips = all_clips.shape[0]
print(f"Total pooled clips: {num_clips}")

# Shuffle
np.random.seed(42)
indices = np.random.permutation(num_clips)
all_clips = all_clips[indices]
all_labels = all_labels[indices]

# Split 70 / 15 / 15
# 662 * 0.7 = 463
# 662 * 0.15 = 99
train_end = int(num_clips * 0.7)
val_end = int(num_clips * 0.85)

train_clips = all_clips[:train_end]
train_labels = all_labels[:train_end]
val_clips = all_clips[train_end:val_end]
val_labels = all_labels[train_end:val_end]
test_clips = all_clips[val_end:]
test_labels = all_labels[val_end:]

print(f"Train: {len(train_clips)}, Val: {len(val_clips)}, Test: {len(test_clips)}")

def write_h5(path, clips, labels):
    with h5py.File(path, "w") as f:
        f.create_dataset("clips", data=clips, compression="gzip")
        f.create_dataset("labels", data=labels, dtype="i8")
        f.attrs["label_map"] = label_map

out_train = "preprocessed_v4/exports/same_signer_train.h5"
out_val = "preprocessed_v4/exports/same_signer_val.h5"
out_test = "preprocessed_v4/exports/same_signer_test.h5"

write_h5(out_train, train_clips, train_labels)
write_h5(out_val, val_clips, val_labels)
write_h5(out_test, test_clips, test_labels)

# 3. Train using baseline
sys.path.append('training')
from train import train_model
from dataset import get_dataloaders
from model import SignLanguageModel

save_dir = "training/logs/same_signer_upper_bound"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "best_model.pth")

device = "cpu"
if torch.backends.mps.is_available(): device = "mps"

start_time = time.time()

# Run baseline training loop (100 epochs, default hyperparameters)
best_val_acc = train_model(
    train_h5=out_train,
    val_h5=out_val,
    num_epochs=100,
    batch_size=32,
    lr=1e-3,
    device=device,
    save_path=save_path
)

training_time = (time.time() - start_time) / 60.0

# Evaluate on test set
num_classes = len(json.loads(label_map))
model = SignLanguageModel(input_size=378, hidden_size=256, num_classes=num_classes, dropout=0.5).to(device)
checkpoint = torch.load(save_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# We can hack get_dataloaders to load test_h5 twice
_, test_loader = get_dataloaders(out_test, out_test, batch_size=32)

test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

test_acc = 100. * test_correct / test_total if test_total > 0 else 0.0

res = {
    "git_hash": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
    "train_h5": "same_signer_train.h5",
    "val_h5": "same_signer_val.h5",
    "test_h5": "same_signer_test.h5",
    "best_epoch": -1, # train_model doesn't return best_epoch directly
    "total_epochs": 100,
    "best_val_acc": best_val_acc,
    "test_acc": test_acc,
    "training_time_minutes": training_time
}

with open(os.path.join(save_dir, "results.json"), "w") as f:
    json.dump(res, f, indent=2)

print("\n=== SAME SIGNER UPPER BOUND RESULTS ===")
print(json.dumps(res, indent=2))
