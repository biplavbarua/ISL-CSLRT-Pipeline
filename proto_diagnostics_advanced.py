import os
import torch
import torch.nn as nn
import sys
sys.path.append('training')
from dataset import ISLDataset
from train_proto import PrototypeModel, euclidean_dist
import numpy as np

def pad_batch(tensors):
    max_len = max([t.shape[0] for t in tensors])
    padded = []
    for t in tensors:
        p = torch.zeros(max_len, t.shape[1])
        p[:t.shape[0], :] = t
        padded.append(p)
    return torch.stack(padded)

def run_diagnostics():
    train_h5 = "preprocessed_v4/exports/standard_train.h5"
    test_h5 = "preprocessed_v4/exports/standard_test.h5"
    model_path = "training/logs/prototype_v1/best_model.pth"
    
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    
    model = PrototypeModel(input_size=378, hidden_size=256, embed_size=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    ds_train = ISLDataset(train_h5, augment=False)
    ds_test = ISLDataset(test_h5, augment=False)
    
    # 1. Precompute embeddings for all train data to save time
    train_xs = []
    train_ys = []
    for i in range(len(ds_train)):
        train_xs.append(ds_train[i][0])
        train_ys.append(ds_train[i][1].item())
        
    class_to_train_idxs = {}
    for i, y in enumerate(train_ys):
        if y not in class_to_train_idxs:
            class_to_train_idxs[y] = []
        class_to_train_idxs[y].append(i)
        
    with torch.no_grad():
        # Batch compute all train embeddings
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=32, collate_fn=lambda b: (pad_batch([x[0] for x in b]), torch.tensor([x[1] for x in b])))
        train_embs = []
        for x_batch, _ in train_loader:
            embs = model(x_batch.to(device))
            train_embs.append(embs.cpu())
        train_embs = torch.cat(train_embs, dim=0) # (N_train, 128)
        
        # Standard Prototypes (mean of all train examples for each class)
        num_classes = len(class_to_train_idxs)
        standard_protos = torch.zeros(num_classes, 128)
        for c in range(num_classes):
            c_idxs = class_to_train_idxs[c]
            c_embs = train_embs[c_idxs]
            proto = c_embs.mean(0)
            proto = nn.functional.normalize(proto, p=2, dim=0)
            standard_protos[c] = proto
            
        # ========================================================
        # TASK 1: TEST SET EVALUATION
        # ========================================================
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=32, collate_fn=lambda b: (pad_batch([x[0] for x in b]), torch.tensor([x[1] for x in b])))
        
        test_intra_dists = []
        test_inter_dists = []
        
        for x_batch, y_batch in test_loader:
            embs = model(x_batch.to(device)).cpu()
            dists = euclidean_dist(embs, standard_protos) # (batch, num_classes)
            
            for i in range(embs.size(0)):
                true_c = y_batch[i].item()
                # Intra-class dist
                test_intra_dists.append(dists[i, true_c].item())
                # Inter-class dist (mean of all OTHER classes)
                mask = torch.ones(num_classes, dtype=torch.bool)
                mask[true_c] = False
                test_inter_dists.append(dists[i, mask].mean().item())
                
        # ========================================================
        # TASK 2: TRAIN SET EVALUATION (LEAVE-ONE-OUT)
        # ========================================================
        train_intra_dists = []
        train_inter_dists = []
        
        for i in range(train_embs.size(0)):
            true_c = train_ys[i]
            c_idxs = class_to_train_idxs[true_c]
            
            if len(c_idxs) < 2:
                # Cannot do leave-one-out if it's the only example
                continue
                
            # Leave-one-out prototype
            other_idxs = [idx for idx in c_idxs if idx != i]
            loo_proto = train_embs[other_idxs].mean(0)
            loo_proto = nn.functional.normalize(loo_proto, p=2, dim=0)
            
            q_emb = train_embs[i:i+1] # (1, 128)
            
            # Intra-class dist to leave-one-out prototype
            intra = euclidean_dist(q_emb, loo_proto.unsqueeze(0))[0, 0].item()
            train_intra_dists.append(intra)
            
            # Inter-class dist to ALL OTHER standard prototypes
            # (since this sample wasn't part of other classes' prototypes anyway)
            mask = torch.ones(num_classes, dtype=torch.bool)
            mask[true_c] = False
            other_protos = standard_protos[mask] # (num_classes-1, 128)
            inter_dists = euclidean_dist(q_emb, other_protos)[0] # (num_classes-1)
            train_inter_dists.append(inter_dists.mean().item())

    # OUTPUT RESULTS
    print("=== TASK 1: TEST SET DIAGNOSTICS ===")
    print(f"Evaluated on: {len(test_intra_dists)} clips")
    print(f"Mean intra-class distance (to TRUE prototype): {np.mean(test_intra_dists):.4f}")
    print(f"Mean inter-class distance (to OTHER prototypes): {np.mean(test_inter_dists):.4f}")
    
    print("\n=== TASK 2: TRAIN SET DIAGNOSTICS (LEAVE-ONE-OUT) ===")
    print(f"Evaluated on: {len(train_intra_dists)} clips (classes with >= 2 examples)")
    print(f"Mean intra-class distance (to LOO true prototype): {np.mean(train_intra_dists):.4f}")
    print(f"Mean inter-class distance (to OTHER prototypes): {np.mean(train_inter_dists):.4f}")

if __name__ == '__main__':
    run_diagnostics()
