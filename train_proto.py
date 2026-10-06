import os
import time
import json
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys

sys.path.append('training')
from dataset import ISLDataset

class PrototypeModel(nn.Module):
    def __init__(self, input_size=378, hidden_size=256, embed_size=128, num_gru_layers=2, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=True
        )
        self.embedding_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, embed_size)
        )
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        embeddings = self.embedding_head(out)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def train_prototypical():
    train_h5 = "preprocessed_v4/exports/standard_train.h5"
    val_h5 = "preprocessed_v4/exports/standard_val.h5"
    test_h5 = "preprocessed_v4/exports/standard_test.h5"
    save_dir = "training/logs/prototype_v1"
    os.makedirs(save_dir, exist_ok=True)
    
    ds_train = ISLDataset(train_h5, augment=True)
    class_to_idx = {}
    for i in range(len(ds_train)):
        x, y = ds_train[i]
        y_val = y.item()
        if y_val not in class_to_idx:
            class_to_idx[y_val] = []
        class_to_idx[y_val].append(i)
        
    valid_classes = [c for c, idxs in class_to_idx.items() if len(idxs) >= 2]
    
    ds_val = ISLDataset(val_h5, augment=False)
    ds_test = ISLDataset(test_h5, augment=False)
    
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    
    model = PrototypeModel(input_size=378, hidden_size=256, embed_size=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    
    best_val_acc = 0.0
    best_epoch = -1
    patience = 15
    epochs_no_improve = 0
    start_time = time.time()
    
    episodes_per_epoch = 50
    ways = min(32, len(valid_classes))
    
    metrics = []
    
    def pad_batch(tensors):
        max_len = max([t.shape[0] for t in tensors])
        padded = []
        for t in tensors:
            p = torch.zeros(max_len, t.shape[1])
            p[:t.shape[0], :] = t
            padded.append(p)
        return torch.stack(padded)
                
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for _ in range(episodes_per_epoch):
            sampled_classes = np.random.choice(valid_classes, ways, replace=False)
            support_xs = []
            query_xs = []
            
            for c in sampled_classes:
                idxs = np.random.choice(class_to_idx[c], 2, replace=False)
                support_xs.append(ds_train[idxs[0]][0])
                query_xs.append(ds_train[idxs[1]][0])
                
            support_batch = pad_batch(support_xs).to(device)
            query_batch = pad_batch(query_xs).to(device)
            
            optimizer.zero_grad()
            emb_s = model(support_batch)
            emb_q = model(query_batch)
            
            dists = euclidean_dist(emb_q, emb_s)
            target = torch.arange(ways).to(device)
            
            loss = nn.CrossEntropyLoss()(-dists, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = (-dists).max(1)
            train_acc += (preds == target).float().mean().item() * 100.0
            
        scheduler.step()
        train_loss /= episodes_per_epoch
        train_acc /= episodes_per_epoch
        
        model.eval()
        with torch.no_grad():
            all_prototypes = {}
            for c in class_to_idx.keys():
                xs = [ds_train[i][0] for i in class_to_idx[c]]
                x_batch = pad_batch(xs).to(device)
                embs = model(x_batch)
                proto = embs.mean(0)
                proto = nn.functional.normalize(proto, p=2, dim=0)
                all_prototypes[c] = proto
                
            num_classes = len(all_prototypes)
            proto_tensor = torch.zeros(num_classes, 128).to(device)
            for c in range(num_classes):
                proto_tensor[c] = all_prototypes[c]
                
            val_correct = 0
            val_loss = 0.0
            
            val_loader = torch.utils.data.DataLoader(ds_val, batch_size=32, collate_fn=lambda b: (pad_batch([x[0] for x in b]), torch.tensor([x[1] for x in b])))
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                embs = model(x_batch)
                dists = euclidean_dist(embs, proto_tensor)
                
                loss = nn.CrossEntropyLoss()(-dists, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
                _, preds = (-dists).max(1)
                val_correct += (preds == y_batch).sum().item()
                
            val_loss /= len(ds_val)
            val_acc_epoch = 100. * val_correct / len(ds_val)
            
        metrics.append({
            "epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc_epoch
        })
        
        is_best = val_acc_epoch > best_val_acc
        if is_best:
            best_val_acc = val_acc_epoch
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    model.eval()
    with torch.no_grad():
        all_prototypes = {}
        for c in class_to_idx.keys():
            xs = [ds_train[i][0] for i in class_to_idx[c]]
            x_batch = pad_batch(xs).to(device)
            embs = model(x_batch)
            proto = embs.mean(0)
            proto = nn.functional.normalize(proto, p=2, dim=0)
            all_prototypes[c] = proto
        proto_tensor = torch.zeros(len(all_prototypes), 128).to(device)
        for c in range(len(all_prototypes)):
            proto_tensor[c] = all_prototypes[c]
            
        test_correct = 0
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=32, collate_fn=lambda b: (pad_batch([x[0] for x in b]), torch.tensor([x[1] for x in b])))
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            embs = model(x_batch)
            dists = euclidean_dist(embs, proto_tensor)
            _, preds = (-dists).max(1)
            test_correct += (preds == y_batch).sum().item()
        test_acc = 100. * test_correct / len(ds_test)
        
    import subprocess
    res = {
        "git_hash": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        "train_h5": "standard_train.h5",
        "val_h5": "standard_val.h5",
        "test_h5": "standard_test.h5",
        "best_epoch": best_epoch,
        "total_epochs": 100,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "training_time_minutes": (time.time() - start_time)/60.0
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    
    print("\n=== PROTOTYPE TRAINING COMPLETE ===")
    print(json.dumps(res, indent=2))
    print("\n=== METRICS PREVIEW ===")
    print(pd.DataFrame(metrics).head(5).to_string(index=False))

if __name__ == '__main__':
    train_prototypical()
