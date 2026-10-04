import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import h5py
import json
import uuid

from dataset import get_dataloaders, ISLDataset
from model import SignLanguageModel
from train import train_model, validate_label_maps
from torch.utils.data import DataLoader

def evaluate_on_test(test_h5, model_path, num_classes, device="cpu"):
    """Evaluates a saved model on a test HDF5 file, verifying its metadata."""
    test_dataset = ISLDataset(test_h5, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "num_classes" not in checkpoint or "label_map" not in checkpoint:
        raise ValueError("Checkpoint is missing metadata. Cannot silently reuse old checkpoints.")
        
    if checkpoint["num_classes"] != num_classes:
        raise ValueError(f"Checkpoint num_classes ({checkpoint['num_classes']}) mismatch with test set ({num_classes})")
        
    with h5py.File(test_h5, 'r') as hf:
        test_label_map_str = hf.attrs['label_map']
        if checkpoint["label_map"] != test_label_map_str:
            raise ValueError("Checkpoint label_map strictly differs from test set label_map.")
    
    model = SignLanguageModel(input_size=378, hidden_size=256, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100. * correct / total

def run_loso_evaluation(exports_dir, epochs=50, device="cpu"):
    print("=" * 60)
    print("STARTING LEAVE-ONE-SIGNER-OUT (LOSO) CROSS-VALIDATION")
    print("=" * 60)
    
    fold_accuracies = []
    
    for fold in range(1, 8):
        print(f"\n--- FOLD {fold} (Test Signer: {fold}) ---")
        train_h5 = os.path.join(exports_dir, f"loso_fold_{fold}_train.h5")
        val_h5   = os.path.join(exports_dir, f"loso_fold_{fold}_val.h5")
        test_h5  = os.path.join(exports_dir, f"loso_fold_{fold}_test.h5")
        
        if not os.path.exists(train_h5):
            print(f"Skipping fold {fold} - missing data files.")
            continue
            
        num_classes = validate_label_maps(train_h5, val_h5, test_h5)
        print(f"Validated {num_classes} identical classes and valid ranges across train/val/test")
            
        run_uuid = uuid.uuid4().hex[:8]
        model_save_path = f"best_model_fold_{fold}_{run_uuid}.pth"
        
        # 1. Train the model (it saves the best epoch to model_save_path)
        # We validate on the 'val' split to determine the best epoch
        print(f"Training...")
        train_model(
            train_h5=train_h5,
            val_h5=val_h5,
            num_epochs=epochs,
            batch_size=32,
            lr=1e-3,
            device=device,
            save_path=model_save_path
        )
        
        # 2. Evaluate the best model on the true unseen test signer
        res = "low-res" if fold in [1, 2] else "HD"
        print(f"Evaluating on Test Signer {fold} ({res})...")
        test_acc = evaluate_on_test(test_h5, model_save_path, num_classes, device=device)
        print(f"Fold {fold} ({res}) Test Accuracy: {test_acc:.2f}%")
        
        fold_accuracies.append((res, test_acc))
        
    print("\n" + "=" * 60)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("=" * 60)
    for i, (res, acc) in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.2f}% ({res})")
        
    all_acc = [acc for res, acc in fold_accuracies]
    hd_acc = [acc for res, acc in fold_accuracies if res == "HD"]
    low_res_acc = [acc for res, acc in fold_accuracies if res == "low-res"]
    
    if hd_acc:
        print(f"\nHD TEST ACCURACY (Folds 3-7): {np.mean(hd_acc):.2f}% ± {np.std(hd_acc):.2f}%")
    if low_res_acc:
        print(f"LOW-RES TEST ACCURACY (Folds 1-2): {np.mean(low_res_acc):.2f}% ± {np.std(low_res_acc):.2f}%")
        
    avg_acc = np.mean(all_acc)
    std_acc = np.std(all_acc)
    print(f"\nMIXED-RESOLUTION POOLED ACCURACY: {avg_acc:.2f}% ± {std_acc:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOSO Evaluation")
    parser.add_argument("--exports_dir", type=str, required=True, help="Directory with HDF5 files")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or mps")
    
    args = parser.parse_args()
    
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
        
    run_loso_evaluation(args.exports_dir, args.epochs, args.device)
