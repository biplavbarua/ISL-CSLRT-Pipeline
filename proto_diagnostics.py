import os
import torch
import torch.nn as nn
import sys
sys.path.append('training')
from dataset import ISLDataset
from train_proto import PrototypeModel, euclidean_dist
import numpy as np

def run_diagnostics():
    train_h5 = "preprocessed_v4/exports/standard_train.h5"
    test_h5 = "preprocessed_v4/exports/standard_test.h5"
    model_path = "training/logs/prototype_v1/best_model.pth"
    
    # --- 1. Class Counts ---
    ds_train = ISLDataset(train_h5, augment=False)
    class_to_idx = {}
    for i in range(len(ds_train)):
        y_val = ds_train[i][1].item()
        if y_val not in class_to_idx:
            class_to_idx[y_val] = []
        class_to_idx[y_val].append(i)
        
    valid_classes = [c for c, idxs in class_to_idx.items() if len(idxs) >= 2]
    
    print("=== 1. TRAINING CLASSES EXCLUDED ===")
    print(f"Total classes in training set: {len(class_to_idx)}")
    print(f"Classes with >= 2 examples (participating): {len(valid_classes)}")
    print(f"Classes with 1 example (excluded from episodes): {len(class_to_idx) - len(valid_classes)}")
    
    # --- 2. Distance Ratios ---
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    
    model = PrototypeModel(input_size=378, hidden_size=256, embed_size=128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    def pad_batch(tensors):
        max_len = max([t.shape[0] for t in tensors])
        padded = []
        for t in tensors:
            p = torch.zeros(max_len, t.shape[1])
            p[:t.shape[0], :] = t
            padded.append(p)
        return torch.stack(padded)
        
    with torch.no_grad():
        # Compute all 97 prototypes
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
            
        # Evaluate on test set
        ds_test = ISLDataset(test_h5, augment=False)
        
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=32, collate_fn=lambda b: (pad_batch([x[0] for x in b]), torch.tensor([x[1] for x in b])))
        
        ratios = []
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            embs = model(x_batch)
            # dists: (batch_size, num_classes)
            dists = euclidean_dist(embs, proto_tensor)
            
            # For each item in batch
            for i in range(embs.size(0)):
                true_c = y_batch[i].item()
                true_dist = dists[i, true_c].item()
                nearest_dist = dists[i].min().item()
                
                # Prevent division by zero just in case
                if true_dist > 1e-6:
                    ratios.append(nearest_dist / true_dist)
                else:
                    # If true_dist is 0, then nearest_dist is also 0, ratio = 1.0
                    ratios.append(1.0)
                    
        print("\n=== 2. DISTANCE RATIO BREAKDOWN ===")
        print(f"Evaluated on {len(ratios)} test clips.")
        print(f"Mean ratio (nearest_distance / true_class_distance): {np.mean(ratios):.4f}")

if __name__ == '__main__':
    run_diagnostics()
