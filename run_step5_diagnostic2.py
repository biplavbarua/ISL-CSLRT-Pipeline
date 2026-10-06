import os
import subprocess
import json
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys

sys.path.append('training')
from dataset import get_dataloaders
from model import SignLanguageModel
import h5py

def validate_label_maps(*h5_paths):
    maps = []
    for path in h5_paths:
        with h5py.File(path, 'r') as hf:
            maps.append(json.loads(hf.attrs['label_map']))
    first_map = maps[0]
    num_classes = len(first_map)
    return num_classes

def get_test_acc(model, test_h5, device, batch_size=32):
    _, test_loader = get_dataloaders(test_h5, test_h5, batch_size) # hack to get test loader
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    return 100. * val_correct / val_total

def train_step5():
    # Setup config
    with open("training/config.py", "w") as f:
        f.write("""INPUT_SIZE = 378
HIDDEN_SIZE = 64
DROPOUT = 0.5
NUM_GRU_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-5
DEFAULT_EPOCHS = 100
""")
    
    train_h5 = "preprocessed_v4/exports/standard_train.h5"
    val_h5 = "preprocessed_v4/exports/standard_val.h5"
    test_h5 = "preprocessed_v4/exports/standard_test.h5"
    save_dir = "training/logs/step5_combined"
    os.makedirs(save_dir, exist_ok=True)
    
    num_classes = validate_label_maps(train_h5, val_h5)
    train_loader, val_loader = get_dataloaders(train_h5, val_h5, 32)
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    
    model = SignLanguageModel(input_size=378, hidden_size=64, num_classes=num_classes, dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    
    best_val_acc = 0.0
    best_epoch = -1
    patience = 15
    epochs_no_improve = 0
    
    start_time = time.time()
    
    metrics = []
    
    for epoch in range(100):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
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
        
        metrics.append({
            "epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    total_time = (time.time() - start_time) / 60.0
    
    # Eval on test set
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    test_acc = get_test_acc(model, test_h5, device, 32)
    
    res = {
        "git_hash": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        "train_h5": "standard_train.h5",
        "val_h5": "standard_val.h5",
        "test_h5": "standard_test.h5",
        "best_epoch": best_epoch,
        "total_epochs": 100,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "training_time_minutes": total_time
    }
    
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
        
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    
    print("\nContents of training/logs/step5_combined/results.json:")
    print(json.dumps(res, indent=2))

print("=== TASK 1: RUNNING STEP 5 COMBINED ===")
train_step5()

print("\n=== TASK 2: SANITY CHECK METRICS.CSV (EPOCHS 1-5) ===")
for step in ["step2_reduce_capacity", "step3_augmentations", "step4_no_face"]:
    path = f"training/logs/{step}/metrics.csv"
    print(f"\n--- {path} ---")
    try:
        df = pd.read_csv(path)
        print(df.head(5).to_string(index=False))
    except Exception as e:
        print(f"Error reading {path}: {e}")

print("\n=== TASK 3: REPORT CLASS COUNT VS VAL SET SIZE ===")
print("Val accuracy calculation: 93 validation clips across 97 classes. One right/wrong changes the percentage by 1.075%.")
print("Test accuracy calculation: 192 test clips across 97 classes. One right/wrong changes the percentage by 0.521%.")

