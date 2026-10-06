import os
import subprocess
import json
import pandas as pd

# 1. Reset everything to baseline
subprocess.run(["git", "checkout", "--", "training/train.py", "training/config.py", "training/dataset.py"], check=True)

# 2. Apply Step 1 (Early Stopping)
with open("training/train.py", "r") as f:
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
with open("training/train.py", "w") as f:
    f.write(train_code)

# 3. Apply Step 2 (Reduced Capacity)
with open("training/config.py", "r") as f:
    config_code = f.read()
config_code = config_code.replace("HIDDEN_SIZE = 256", "HIDDEN_SIZE = 64")
with open("training/config.py", "w") as f:
    f.write(config_code)

print("=== TASK 1: RUNNING STEP 5 COMBINED ===")
cmd = [
    "python3", "training/train.py",
    "--train_h5", "../preprocessed_v4/exports/standard_train.h5",
    "--val_h5", "../preprocessed_v4/exports/standard_val.h5",
    "--test_h5", "../preprocessed_v4/exports/standard_test.h5",
    "--save_dir", "training/logs/step5_combined"
]
subprocess.run(cmd, check=True)
print("\nContents of training/logs/step5_combined/results.json:")
with open("training/logs/step5_combined/results.json", "r") as f:
    print(f.read())

print("\n=== TASK 2: SANITY CHECK METRICS.CSV (EPOCHS 1-5) ===")
for step in ["step2_reduce_capacity", "step3_augmentations", "step4_no_face"]:
    path = f"training/logs/{step}/metrics.csv"
    print(f"\n--- {path} ---")
    try:
        df = pd.read_csv(path)
        print(df.head(5).to_string(index=False))
    except Exception as e:
        print(f"Error reading {path}: {e}")

