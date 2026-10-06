import os
import subprocess
import json

def run_train(name):
    print(f"\n===========================================")
    print(f"RUNNING EXPERIMENT: {name}")
    print(f"===========================================\n")
    cmd = [
        "python3", "train.py",
        "--train_h5", "../preprocessed_v4/exports/standard_train.h5",
        "--val_h5", "../preprocessed_v4/exports/standard_val.h5",
        "--test_h5", "../preprocessed_v4/exports/standard_test.h5",
        "--save_dir", f"logs/{name}"
    ]
    subprocess.run(cmd, check=True)
    
    with open(f"logs/{name}/results.json") as f:
        return json.load(f)

def main():
    # Step 1: Add Early Stopping
    with open("train.py", "r") as f:
        train_code = f.read()

    train_code = train_code.replace(
        "best_val_acc = 0.0\n    best_epoch = -1",
        "best_val_acc = 0.0\n    best_epoch = -1\n    patience = 15\n    epochs_no_improve = 0"
    )
    train_code = train_code.replace(
"""        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1""",
"""        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0"""
    )
    train_code = train_code.replace(
"""            }, save_path)
            
        if (epoch + 1) % 5 == 0""",
"""            }, save_path)
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 5 == 0"""
    )
    train_code = train_code.replace(
"""            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%{best_str}")
            
    total_time = time.time() - start_time""",
"""            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%{best_str}")
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    total_time = time.time() - start_time"""
    )

    with open("train.py", "w") as f:
        f.write(train_code)

    res1 = run_train("step1_early_stopping")

    # Step 2: Reduce Model Capacity
    with open("config.py", "r") as f:
        config_code = f.read()
    config_code = config_code.replace("HIDDEN_SIZE = 256", "HIDDEN_SIZE = 64")
    with open("config.py", "w") as f:
        f.write(config_code)

    res2 = run_train("step2_reduce_capacity")

    # Step 3: Strengthen Augmentation
    with open("dataset.py", "r") as f:
        dataset_code = f.read()

    aug_new = """        if self.augment:
            # 1. Spatial jitter
            noise = np.random.normal(loc=0.0, scale=0.01, size=x.shape).astype(np.float32)
            x += noise
            
            # 2. Random rotation around Y axis (vertical)
            angle = np.random.uniform(-15, 15) * np.pi / 180.0
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            x_coords = x[:, :, 0].copy()
            z_coords = x[:, :, 2].copy()
            x[:, :, 0] = x_coords * cos_a + z_coords * sin_a
            x[:, :, 2] = -x_coords * sin_a + z_coords * cos_a
            
            # 3. Temporal frame dropout
            if np.random.rand() < 0.5:
                num_drop = np.random.randint(1, 4)
                start_drop = np.random.randint(0, self.n_frames - num_drop)
                x[start_drop:start_drop+num_drop, :, :] = 0.0"""
    
    dataset_code = dataset_code.replace(
"""        if self.augment:
            # Spatial Augmentation: add slight gaussian noise to coordinates.
            # standard deviation of 0.01 (1% of shoulder width).
            noise = np.random.normal(loc=0.0, scale=0.01, size=x.shape).astype(np.float32)
            x += noise""", aug_new
    )
    with open("dataset.py", "w") as f:
        f.write(dataset_code)

    res3 = run_train("step3_augmentations")

    # Step 4: Ablate input dimensionality
    with open("config.py", "r") as f:
        config_code = f.read()
    config_code = config_code.replace("INPUT_SIZE = 378", "INPUT_SIZE = 225")
    with open("config.py", "w") as f:
        f.write(config_code)

    with open("dataset.py", "r") as f:
        dataset_code = f.read()
    dataset_code = dataset_code.replace(
        "SELECTED_INDICES = POSE_INDICES + LHAND_INDICES + RHAND_INDICES + FACE_INDICES",
        "SELECTED_INDICES = POSE_INDICES + LHAND_INDICES + RHAND_INDICES"
    )
    with open("dataset.py", "w") as f:
        f.write(dataset_code)

    res4 = run_train("step4_no_face")

    # Print table
    print("\n\n=== FINAL RESULTS ===")
    print("Step 1 (Early Stopping):")
    print(f"  Best Val Acc: {res1['best_val_acc']:.2f}% (Epoch {res1['best_epoch']}) | Test Acc: {res1['test_acc']:.2f}%")
    print("Step 2 (Reduce Capacity to 64):")
    print(f"  Best Val Acc: {res2['best_val_acc']:.2f}% (Epoch {res2['best_epoch']}) | Test Acc: {res2['test_acc']:.2f}%")
    print("Step 3 (Strong Augmentations):")
    print(f"  Best Val Acc: {res3['best_val_acc']:.2f}% (Epoch {res3['best_epoch']}) | Test Acc: {res3['test_acc']:.2f}%")
    print("Step 4 (Ablate Face Landmarks):")
    print(f"  Best Val Acc: {res4['best_val_acc']:.2f}% (Epoch {res4['best_epoch']}) | Test Acc: {res4['test_acc']:.2f}%")

if __name__ == "__main__":
    main()
