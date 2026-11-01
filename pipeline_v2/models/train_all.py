"""
pipeline_v2/models/train_all.py

Unified training script for all three models.
Runs on: standard_train.h5 / standard_val.h5 (Step 05 outputs).

Usage:
    cd ISL-CSLRT-Pipeline
    python3 pipeline_v2/models/train_all.py --exports_dir preprocessed_v5/exports --device mps

Results are saved to: pipeline_v2/logs/<model_name>/
    results.json  — best_val_acc, test_acc, training_time_minutes
    metrics.csv   — epoch-by-epoch train/val loss and accuracy
"""

import os
import sys
import json
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import h5py

# Add pipeline_v2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dataset           import ISLDataset, get_dataloaders
from lstm_classifier   import LSTMClassifier
from stgcn             import STGCNClassifier
from pose_transformer  import PoseTransformer

LOGS_ROOT = os.path.join(os.path.dirname(__file__), "..", "logs")


# ─── STGCN needs (B, 3, T, 75) not (B, T, 378) ──────────────────────────────
class STGCNDataset(ISLDataset):
    """Re-shapes (T, 378) → (3, T, 75) for STGCN."""
    def __init__(self, h5_path, augment=False):
        super().__init__(h5_path, augment=augment, flatten=False)
        # clips from ISLDataset are (T, 126, 3); we need pose+hand only (T, 75, 3)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)  # x: (T, 126, 3)
        # Take first 75 nodes: Pose (33) + L-Hand (21) + R-Hand (21)
        x = x[:, :75, :]          # (T, 75, 3)
        x = x.permute(2, 0, 1)    # (3, T, 75)
        return x, y


def get_num_classes(h5_path: str) -> tuple[int, str]:
    with h5py.File(h5_path, "r") as hf:
        label_map_str = hf.attrs["label_map"]
        n_classes = len(json.loads(label_map_str))
    return n_classes, label_map_str


def validate_splits(train_h5: str, val_h5: str, test_h5: str):
    """Confirm label maps are identical across splits."""
    maps = []
    for p in [train_h5, val_h5, test_h5]:
        with h5py.File(p, "r") as hf:
            maps.append(json.loads(hf.attrs["label_map"]))
    if maps[0] != maps[1] or maps[0] != maps[2]:
        raise ValueError("Label map mismatch between splits!")
    return len(maps[0])


def train_one_model(
    model_name:   str,
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_h5:      str,
    n_classes:    int,
    label_map_str:str,
    epochs:       int,
    lr:           float,
    device:       str,
    is_stgcn:     bool = False,
):
    log_dir = os.path.join(LOGS_ROOT, model_name)
    os.makedirs(log_dir, exist_ok=True)

    model = model.to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc  = 0.0
    best_epoch    = -1
    patience      = 15
    no_improve    = 0
    metric_rows   = []
    model_path    = os.path.join(log_dir, "best_model.pth")

    start_time = time.time()

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item() * xb.size(0)
            preds          = out.argmax(1)
            train_correct += (preds == yb).sum().item()
            train_total   += xb.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc   = 100.0 * train_correct / train_total

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out  = model(xb)
                loss = criterion(out, yb)
                val_loss    += loss.item() * xb.size(0)
                preds        = out.argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total   += xb.size(0)

        val_loss /= val_total
        val_acc   = 100.0 * val_correct / val_total

        is_best = (val_acc > best_val_acc) or (epoch == 0)
        if is_best:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch + 1
                no_improve   = 0
            elif epoch == 0:
                best_val_acc = val_acc
                best_epoch   = 1
            torch.save({
                "state_dict":    model.state_dict(),
                "num_classes":   n_classes,
                "label_map":     label_map_str,
                "model_name":    model_name,
            }, model_path)
        else:
            no_improve += 1

        metric_rows.append({
            "epoch": epoch + 1, "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 3), "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 3),
        })

        if (epoch + 1) % 5 == 0 or is_best or epoch == 0:
            tag = " ★" if is_best else ""
            print(f"  [{model_name}] Epoch {epoch+1:3d}/{epochs} | "
                  f"Train {train_acc:.2f}% | Val {val_acc:.2f}%{tag}")

        if no_improve >= patience:
            print(f"  [{model_name}] Early stopping at epoch {epoch+1}")
            break

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print(f"\n  [{model_name}] Loading best checkpoint (epoch {best_epoch}) for test...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if is_stgcn:
        test_ds = STGCNDataset(test_h5, augment=False)
    else:
        test_ds = ISLDataset(test_h5, augment=False, flatten=True)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    test_correct = test_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds  = model(xb).argmax(1)
            test_correct += (preds == yb).sum().item()
            test_total   += xb.size(0)

    test_acc = 100.0 * test_correct / test_total
    elapsed  = (time.time() - start_time) / 60.0

    results = {
        "model":                  model_name,
        "best_epoch":             best_epoch,
        "total_epochs_run":       len(metric_rows),
        "best_val_acc":           round(best_val_acc, 4),
        "test_acc":               round(test_acc, 4),
        "training_time_minutes":  round(elapsed, 3),
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(log_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc"])
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"\n  [{model_name}] DONE — Val={best_val_acc:.2f}%  Test={test_acc:.2f}%  Time={elapsed:.1f}min")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train all 3 ISL-CSLRT models")
    parser.add_argument("--exports_dir", type=str, required=True,
                        help="Directory containing standard_*.h5 files from step05")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per model")
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu | mps | cuda")
    parser.add_argument("--model",  type=str, default="all",
                        choices=["all", "lstm", "stgcn", "transformer"],
                        help="Which model to train (default: all)")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"

    train_h5 = os.path.join(args.exports_dir, "standard_train.h5")
    val_h5   = os.path.join(args.exports_dir, "standard_val.h5")
    test_h5  = os.path.join(args.exports_dir, "standard_test.h5")

    for p in [train_h5, val_h5, test_h5]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing HDF5 file: {p}")

    n_classes, label_map_str = get_num_classes(train_h5)
    validate_splits(train_h5, val_h5, test_h5)
    print(f"Dataset: {n_classes} classes  device={args.device}  epochs={args.epochs}")

    all_results = {}

    # ── Model 1: BiLSTM ───────────────────────────────────────────────────────
    if args.model in ("all", "lstm"):
        print("\n" + "="*60)
        print("MODEL 1: Bi-LSTM Classifier")
        print("="*60)
        train_dl, val_dl = get_dataloaders(train_h5, val_h5, batch_size=32,
                                           augment=True, flatten=True)
        model = LSTMClassifier(input_size=378, hidden_size=256,
                               num_classes=n_classes, dropout=0.5)
        res = train_one_model("lstm", model, train_dl, val_dl, test_h5,
                              n_classes, label_map_str, args.epochs, args.lr,
                              args.device, is_stgcn=False)
        all_results["lstm"] = res

    # ── Model 2: ST-GCN ───────────────────────────────────────────────────────
    if args.model in ("all", "stgcn"):
        print("\n" + "="*60)
        print("MODEL 2: ST-GCN Classifier")
        print("="*60)
        train_ds_stgcn = STGCNDataset(train_h5, augment=True)
        val_ds_stgcn   = STGCNDataset(val_h5,   augment=False)
        train_dl_stgcn = DataLoader(train_ds_stgcn, batch_size=32, shuffle=True)
        val_dl_stgcn   = DataLoader(val_ds_stgcn,   batch_size=32, shuffle=False)
        model_stgcn = STGCNClassifier(in_channels=3, num_classes=n_classes)
        res = train_one_model("stgcn", model_stgcn, train_dl_stgcn, val_dl_stgcn,
                              test_h5, n_classes, label_map_str, args.epochs, args.lr,
                              args.device, is_stgcn=True)
        all_results["stgcn"] = res

    # ── Model 3: Pose Transformer ─────────────────────────────────────────────
    if args.model in ("all", "transformer"):
        print("\n" + "="*60)
        print("MODEL 3: Pose Transformer Classifier")
        print("="*60)
        train_dl, val_dl = get_dataloaders(train_h5, val_h5, batch_size=32,
                                           augment=True, flatten=True)
        model_tf = PoseTransformer(input_size=378, d_model=256, n_heads=8,
                                   n_layers=4, num_classes=n_classes, dropout=0.1)
        res = train_one_model("transformer", model_tf, train_dl, val_dl, test_h5,
                              n_classes, label_map_str, args.epochs, args.lr,
                              args.device, is_stgcn=False)
        all_results["transformer"] = res

    # ── Summary Table ─────────────────────────────────────────────────────────
    if all_results:
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print(f"{'Model':<20} {'Best Val Acc':>14} {'Test Acc':>10} {'Time (min)':>12}")
        print("-"*60)
        for name, r in all_results.items():
            print(f"{r['model']:<20} {r['best_val_acc']:>13.2f}%"
                  f" {r['test_acc']:>9.2f}%  {r['training_time_minutes']:>10.1f}")
        print("="*60)
        print(f"Random chance: {100/n_classes:.2f}%")

        summary_path = os.path.join(LOGS_ROOT, "comparison_summary.json")
        os.makedirs(LOGS_ROOT, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison saved to: {summary_path}")


if __name__ == "__main__":
    main()

