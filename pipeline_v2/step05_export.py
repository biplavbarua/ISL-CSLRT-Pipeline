"""
pipeline_v2/step05_export.py

Step 05 — Pack CSV splits into HDF5 files and generate QA plots.

Each .h5 file contains:
    clips      — (N, T, 126, 3) float32  — 126-landmark subset, T=32 frames
    labels     — (N,)           int32
    signer_ids — (N,)           int32
    glosses    — (N,)           string (sentence slug)

Attributes on the file root:
    label_map  — JSON string: {sentence_slug: label_int}
    n_classes  — int
    split      — str (e.g. "standard_train")
"""

import os
import sys
import csv
import json
import numpy as np
import h5py
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config


def select_landmarks(clip: np.ndarray) -> np.ndarray:
    """
    Reduce (T, 543, 3) → (T, 126, 3) using the 126-landmark subset
    defined in config.SELECTED_INDICES.
    """
    return clip[:, config.SELECTED_INDICES, :]  # (T, 126, 3)


def load_split_csv(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def write_h5(rows: list[dict], h5_path: str, split_name: str, label_map: dict):
    clips_list   = []
    labels_list  = []
    signers_list = []
    glosses_list = []

    skipped = 0
    for row in rows:
        seg_path = row["seg_path"]
        if not os.path.exists(seg_path):
            print(f"  [WARN] Missing segment: {seg_path}")
            skipped += 1
            continue

        clip = np.load(seg_path)          # (T, 543, 3)
        clip = select_landmarks(clip)     # (T, 126, 3)

        clips_list.append(clip)
        labels_list.append(int(row["label"]))
        signers_list.append(int(row["signer_id"]))
        glosses_list.append(row["sentence"])

    if not clips_list:
        print(f"  [ERROR] No valid clips for split {split_name}. Skipping HDF5 write.")
        return

    clips_arr   = np.stack(clips_list,  axis=0).astype(np.float32)  # (N, T, 126, 3)
    labels_arr  = np.array(labels_list, dtype=np.int32)
    signers_arr = np.array(signers_list, dtype=np.int32)

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("clips",      data=clips_arr,   compression="gzip")
        hf.create_dataset("labels",     data=labels_arr)
        hf.create_dataset("signer_ids", data=signers_arr)
        dt = h5py.special_dtype(vlen=str)
        ds = hf.create_dataset("glosses", (len(glosses_list),), dtype=dt)
        for i, g in enumerate(glosses_list):
            ds[i] = g
        hf.attrs["label_map"] = json.dumps(label_map)
        hf.attrs["n_classes"] = len(label_map)
        hf.attrs["split"]     = split_name

    print(f"  [H5] {os.path.basename(h5_path)}: {clips_arr.shape[0]} clips  "
          f"shape={clips_arr.shape}  skipped={skipped}")


def make_qa_plots(label_map: dict):
    """Bar chart of class distribution across standard train/val/test."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    split_names = ["standard_train", "standard_val", "standard_test"]
    split_data  = {}

    for sname in split_names:
        h5p = os.path.join(config.EXPORTS_DIR, f"{sname}.h5")
        if not os.path.exists(h5p):
            continue
        with h5py.File(h5p, "r") as hf:
            labels = hf["labels"][:]
        counts = np.bincount(labels, minlength=len(label_map))
        split_data[sname] = counts

    if not split_data:
        return

    fig, axes = plt.subplots(len(split_data), 1,
                              figsize=(14, 3 * len(split_data)), squeeze=False)
    for ax, (sname, counts) in zip(axes[:, 0], split_data.items()):
        ax.bar(range(len(counts)), counts)
        ax.set_title(f"{sname} — class distribution")
        ax.set_xlabel("Label index")
        ax.set_ylabel("Clip count")
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "dataset_qa.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  [QA] Plot saved: {out}")


def process_all():
    os.makedirs(config.EXPORTS_DIR, exist_ok=True)

    label_map_path = os.path.join(config.LOGS_DIR, "label_map.json")
    if not os.path.exists(label_map_path):
        print(f"ERROR: label_map.json not found at {label_map_path}")
        print("  → Run step04_build_splits.py first.")
        return

    with open(label_map_path) as f:
        label_map = json.load(f)

    splits_dir = config.SPLITS_DIR
    if not os.path.isdir(splits_dir):
        print(f"ERROR: splits/ not found at {splits_dir}")
        return

    csv_files = [f for f in sorted(os.listdir(splits_dir)) if f.endswith(".csv")]
    for csv_file in csv_files:
        split_name = csv_file.replace(".csv", "")
        csv_path   = os.path.join(splits_dir, csv_file)
        h5_path    = os.path.join(config.EXPORTS_DIR, f"{split_name}.h5")

        print(f"\nExporting: {split_name}")
        rows = load_split_csv(csv_path)
        write_h5(rows, h5_path, split_name, label_map)

    make_qa_plots(label_map)
    print(f"\nStep 05 complete. HDF5 files in: {config.EXPORTS_DIR}")


if __name__ == "__main__":
    process_all()

