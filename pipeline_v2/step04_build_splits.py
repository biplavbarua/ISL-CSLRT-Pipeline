"""
pipeline_v2/step04_build_splits.py

Step 04 — Build signer-independent dataset splits.

Standard Split (same as old pipeline):
    Train  → Signers 3, 4, 5, 6  (HD studio)
    Val    → Signer 7             (HD studio, unseen during train)
    Test   → Signers 1, 2         (mobile, unseen domain)

Additionally generates 7 Leave-One-Signer-Out (LOSO) folds.

Reads  : preprocessed_v5/segments/<slug>/<signer>/segment.npy
Writes : preprocessed_v5/splits/*.csv
         preprocessed_v5/logs/label_map.json
         preprocessed_v5/logs/split_summary.json
"""

import os
import sys
import csv
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config

ALL_SIGNERS = list(range(1, 8))  # 1..7


def collect_clips() -> list[dict]:
    """Walk segments/ and return a list of clip records with signer detection."""
    clips = []
    seg_root = config.SEG_DIR

    if not os.path.isdir(seg_root):
        raise FileNotFoundError(f"Segments directory not found: {seg_root}")

    for sent_slug in sorted(os.listdir(seg_root)):
        sent_dir = os.path.join(seg_root, sent_slug)
        if not os.path.isdir(sent_dir):
            continue

        for signer_str in sorted(os.listdir(sent_dir)):
            seg_path = os.path.join(sent_dir, signer_str, "segment.npy")
            if not os.path.exists(seg_path):
                continue

            # Verify the segment has actual non-zero content
            seg = np.load(seg_path)
            if seg.max() == 0.0 and seg.min() == 0.0:
                print(f"  [SKIP] Zero-content segment: {sent_slug}/{signer_str}")
                continue

            try:
                signer_id = int(signer_str)
            except ValueError:
                print(f"  [WARN] Non-integer signer folder: {sent_slug}/{signer_str}")
                continue

            clips.append({
                "sentence":  sent_slug,
                "signer_id": signer_id,
                "seg_path":  seg_path,
            })

    return clips


def build_label_map(clips: list[dict]) -> dict[str, int]:
    """Assign a contiguous integer label to each unique sentence slug."""
    sentences = sorted(set(c["sentence"] for c in clips))
    return {s: i for i, s in enumerate(sentences)}


def write_split(rows: list[dict], path: str):
    """Write a CSV split file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["sentence", "signer_id", "label", "seg_path"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clips_to_rows(clips: list[dict], label_map: dict[str, int]) -> list[dict]:
    return [
        {
            "sentence":  c["sentence"],
            "signer_id": c["signer_id"],
            "label":     label_map[c["sentence"]],
            "seg_path":  c["seg_path"],
        }
        for c in clips
    ]


def process_all():
    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR,   exist_ok=True)

    print("Collecting clips from segments/...")
    clips = collect_clips()
    print(f"  Found {len(clips)} valid clips.")

    label_map = build_label_map(clips)
    n_classes = len(label_map)
    print(f"  {n_classes} unique sentence classes.")

    # Save label map
    with open(os.path.join(config.LOGS_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)

    # ── Standard Split ────────────────────────────────────────────────────────
    train_clips = [c for c in clips if c["signer_id"] in config.TRAIN_SIGNERS]
    val_clips   = [c for c in clips if c["signer_id"] in config.VAL_SIGNERS]
    test_clips  = [c for c in clips if c["signer_id"] in config.TEST_SIGNERS]

    write_split(clips_to_rows(train_clips, label_map),
                os.path.join(config.SPLITS_DIR, "standard_train.csv"))
    write_split(clips_to_rows(val_clips, label_map),
                os.path.join(config.SPLITS_DIR, "standard_val.csv"))
    write_split(clips_to_rows(test_clips, label_map),
                os.path.join(config.SPLITS_DIR, "standard_test.csv"))

    print(f"\n  Standard split:  Train={len(train_clips)}  Val={len(val_clips)}  Test={len(test_clips)}")

    # ── LOSO Folds ────────────────────────────────────────────────────────────
    loso_summary = {}
    for test_signer in ALL_SIGNERS:
        val_signer   = (test_signer % 7) + 1  # simple round-robin val signer
        train_s      = [s for s in ALL_SIGNERS if s != test_signer and s != val_signer]

        fold_train = [c for c in clips if c["signer_id"] in train_s]
        fold_val   = [c for c in clips if c["signer_id"] == val_signer]
        fold_test  = [c for c in clips if c["signer_id"] == test_signer]

        write_split(clips_to_rows(fold_train, label_map),
                    os.path.join(config.SPLITS_DIR, f"loso_fold_{test_signer}_train.csv"))
        write_split(clips_to_rows(fold_val, label_map),
                    os.path.join(config.SPLITS_DIR, f"loso_fold_{test_signer}_val.csv"))
        write_split(clips_to_rows(fold_test, label_map),
                    os.path.join(config.SPLITS_DIR, f"loso_fold_{test_signer}_test.csv"))

        loso_summary[f"fold_{test_signer}"] = {
            "test_signer": test_signer,
            "val_signer":  val_signer,
            "train_signers": train_s,
            "train_size": len(fold_train),
            "val_size":   len(fold_val),
            "test_size":  len(fold_test),
        }
        print(f"  LOSO Fold {test_signer}: Train={len(fold_train)}  Val={len(fold_val)}  Test={len(fold_test)}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "total_clips": len(clips),
        "n_classes":   n_classes,
        "standard": {
            "train": len(train_clips),
            "val":   len(val_clips),
            "test":  len(test_clips),
        },
        "loso": loso_summary,
    }
    with open(os.path.join(config.LOGS_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nStep 04 complete. Splits written to {config.SPLITS_DIR}")


if __name__ == "__main__":
    process_all()

