import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os

from dataset import get_dataloaders
from model import SignLanguageModel

import h5py
import json
import numpy as np

def validate_label_maps(*h5_paths):
    """
    Validates that all provided HDF5 files have identical class-name-to-ID maps,
    and that all labels inside their datasets fall within the valid [0, num_classes-1] range.
    Returns the number of classes.
    """
    maps = []
    for path in h5_paths:
        with h5py.File(path, 'r') as hf:
            if 'label_map' not in hf.attrs:
                raise ValueError(f"Missing 'label_map' attribute in {path}")
            maps.append(json.loads(hf.attrs['label_map']))
    
    first_map = maps[0]
    for i, m in enumerate(maps[1:], 1):
        if m != first_map:
            raise ValueError(f"Label map mismatch between {h5_paths[0]} and {h5_paths[i]}")
            
    num_classes = len(first_map)
    ids = sorted(list(first_map.values()))
    if ids != list(range(num_classes)):
        raise ValueError(f"Label IDs are not contiguous 0 to {num_classes-1}")
    
    # Check label ranges
    for path in h5_paths:
        with h5py.File(path, 'r') as hf:
            labels = hf['labels'][:]
            if len(labels) == 0:
                raise ValueError(f"Split {path} is empty! Cannot train/evaluate on empty dataset.")
            min_l, max_l = int(np.min(labels)), int(np.max(labels))
            if min_l < 0 or max_l >= num_classes:
                raise ValueError(f"Label range out of bounds in {path}: min={min_l}, max={max_l}, num_classes={num_classes}")
                
    return num_classes

def train_model(train_h5, val_h5, num_epochs=100, batch_size=32, lr=1e-3, device="cpu", save_path="best_model.pth"):
    num_classes = validate_label_maps(train_h5, val_h5)
    with h5py.File(train_h5, 'r') as hf:
        label_map_str = hf.attrs['label_map']
        
    print(f"Loading data... (validated {num_classes} identical classes and valid ranges across train/val)")
    train_loader, val_loader = get_dataloaders(train_h5, val_h5, batch_size)
    print(f"Train size: {len(train_loader.dataset)} clips | Val size: {len(val_loader.dataset)} clips")
    
    model = SignLanguageModel(input_size=378, hidden_size=256, num_classes=num_classes, dropout=0.5).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Decays learning rate following a cosine curve towards 1e-5
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    best_val_acc = 0.0
    
    print(f"Starting training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        # -- Training Phase --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        # Print progress every 5 epochs or if it's the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "label_map": label_map_str
            }, save_path)
            
        if (epoch + 1) % 5 == 0 or is_best or epoch == 0:
            best_str = " (New Best!)" if is_best else ""
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%{best_str}")
            
    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL-CSLRT Recognition Model")
    parser.add_argument("--train_h5", type=str, required=True, help="Path to training HDF5 file")
    parser.add_argument("--val_h5", type=str, required=True, help="Path to validation HDF5 file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save best model")
    
    args = parser.parse_args()
    
    # Automatically use MPS (Apple Silicon GPU) if available and requested device is mps or auto
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
        
    train_model(
        train_h5=args.train_h5,
        val_h5=args.val_h5,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path
    )
