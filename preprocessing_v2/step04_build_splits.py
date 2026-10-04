"""
step04_build_splits.py — Signer-Independent Data Splits
=========================================================

PURPOSE
-------
Divide the dataset into train / validation / test sets in a way that guarantees
no signer appears in more than one set. This is the research gap: every
published paper on ISL-CSLRT has mixed signers between splits.

THE PROBLEM WITH RANDOM SPLITS
--------------------------------
If you randomly assign 80% of frames to train and 20% to test, you will have
clips from Signer 3 in BOTH sets. The model will learn Signer 3's style —
their particular hand size, posture, and gesture speed. On test, it recognises
Signer 3 by style alone and appears 95% accurate.

Deploy it on a new person and it drops to 40–60%. That performance cliff is
exactly the signer-dependency problem. It is the #1 reason ISL recognition
systems fail in the real world.

LEAVE-ONE-SIGNER-OUT (LOSO) CROSS-VALIDATION
----------------------------------------------
We create 7 folds. In fold K:
  - Test  set = ALL clips from Signer K
  - Val   set = ALL clips from Signer (K mod 7)+1  (next signer, wraps around)
  - Train set = ALL clips from remaining 5 signers

This way:
  1. The model NEVER sees the test signer during training
  2. We evaluate on 7 different "unseen signers" across 7 folds
  3. The average accuracy across folds is a honest estimate of real-world
     generalisation — the metric no paper has ever reported for this dataset

STANDARD SPLIT (for quick experiments)
----------------------------------------
Training full LOSO (7 folds × full training run) is expensive. For initial
experiments and ablations, we also create one fixed split:
  Train = Signers 3, 4, 5, 6  (HD, complete data)
  Val   = Signer 7             (HD, slightly incomplete — realistic)
  Test  = Signers 1 & 2       (low-res — the hardest generalisation challenge)

This is deliberately the hardest possible test: the model trains on HD data
and must recognise low-resolution mobile footage from new signers.

OUTPUT FORMAT
-------------
All splits saved to splits/ as CSV files:
  standard_train.csv, standard_val.csv, standard_test.csv
  loso_fold_1_train.csv, loso_fold_1_val.csv, loso_fold_1_test.csv
  ... (7 folds)

Each CSV contains rows from the segment_manifest.csv with the same columns
plus a "split" column. You can pass these CSVs directly to a PyTorch Dataset.

Also saves a summary JSON with per-split statistics.
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
os.makedirs(config.LOGS_DIR,  exist_ok=True)
os.makedirs(config.SPLITS_DIR, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    """Load the segment manifest from Step 03."""
    manifest_path = os.path.join(config.LOGS_DIR, "segment_manifest.csv")
    if not os.path.exists(manifest_path):
        log.error(f"Segment manifest not found: {manifest_path}")
        log.error("Run step03_segment.py first.")
        sys.exit(1)
    with open(manifest_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
        
    rows = []
    # Convert signer_id to int for filtering, and skip excluded clips
    for r in all_rows:
        clip_id = f"{r['sentence_slug']}/{r['signer_id']}"
        if hasattr(config, 'EXCLUDED_CLIPS') and clip_id in config.EXCLUDED_CLIPS:
            log.info(f"Skipping excluded clip: {clip_id}")
            continue
        r["signer_id"] = int(r["signer_id"])
        rows.append(r)
        
    log.info(f"Manifest loaded: {len(rows)} valid clips (excluded {len(all_rows) - len(rows)})")
    return rows


def save_split(rows: list[dict], split_name: str, split_label: str):
    """Write a list of clip rows to a CSV file in splits/."""
    path = os.path.join(config.SPLITS_DIR, f"{split_name}.csv")
    if not rows:
        log.warning(f"  Empty split '{split_name}' — not saved.")
        return

    fieldnames = list(rows[0].keys()) + ["split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": split_label})

    log.info(f"  Saved {len(rows):4d} clips → {os.path.basename(path)}")


def split_stats(rows: list[dict], name: str) -> dict:
    """Compute per-split statistics for the summary report."""
    signers    = sorted(set(r["signer_id"] for r in rows))
    sentences  = sorted(set(r["sentence_slug"] for r in rows))
    low_res    = [r for r in rows if r["resolution"] == "low_res"]
    return {
        "name":           name,
        "n_clips":        len(rows),
        "n_sentences":    len(sentences),
        "n_signers":      len(signers),
        "signers":        signers,
        "n_low_res":      len(low_res),
        "n_hd":           len(rows) - len(low_res),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_splits():
    fh = logging.FileHandler(os.path.join(config.LOGS_DIR, "step04_splits.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)
    try:
        log.info("=" * 70)
        log.info("ISL-CSLRT PREPROCESSING  —  Step 04: Build Signer-Independent Splits")
        log.info("=" * 70)

        manifest = load_manifest()

        # Index by signer_id for fast lookup
        by_signer: dict[int, list[dict]] = defaultdict(list)
        for row in manifest:
            by_signer[row["signer_id"]].append(row)

        log.info(f"\n  Clips per signer:")
        for s in sorted(by_signer.keys()):
            n = len(by_signer[s])
            res = "low_res" if s in config.LOW_RES_SIGNERS else "HD"
            log.info(f"    Signer {s}: {n} clips  [{res}]")

        summary = {}

        # -----------------------------------------------------------------------
        # STANDARD SPLIT
        # -----------------------------------------------------------------------
        log.info("\n[STANDARD SPLIT]  Train=3,4,5,6  |  Val=7  |  Test=1,2")

        std = config.STANDARD_SPLIT
        std_train = [r for r in manifest if r["signer_id"] in std["train"]]
        std_val   = [r for r in manifest if r["signer_id"] in std["val"]]
        std_test  = [r for r in manifest if r["signer_id"] in std["test"]]

        save_split(std_train, "standard_train", "train")
        save_split(std_val,   "standard_val",   "val")
        save_split(std_test,  "standard_test",  "test")

        summary["standard"] = {
            "train": split_stats(std_train, "standard_train"),
            "val":   split_stats(std_val,   "standard_val"),
            "test":  split_stats(std_test,  "standard_test"),
        }

        # -----------------------------------------------------------------------
        # LOSO FOLDS
        # -----------------------------------------------------------------------
        log.info(f"\n[LOSO CROSS-VALIDATION]  7 folds")

        for fold_name, fold_cfg in config.LOSO_FOLDS.items():
            test_signer  = fold_cfg["test"]
            val_signer   = fold_cfg["val"]
            train_signers = fold_cfg["train"]

            fold_train = [r for r in manifest if r["signer_id"] in train_signers]
            fold_val   = [r for r in manifest if r["signer_id"] == val_signer]
            fold_test  = [r for r in manifest if r["signer_id"] == test_signer]

            log.info(f"\n  {fold_name}: test=Signer{test_signer}, val=Signer{val_signer}, train=Signers{train_signers}")
            save_split(fold_train, f"loso_{fold_name}_train", "train")
            save_split(fold_val,   f"loso_{fold_name}_val",   "val")
            save_split(fold_test,  f"loso_{fold_name}_test",  "test")

            summary[fold_name] = {
                "train": split_stats(fold_train, f"loso_{fold_name}_train"),
                "val":   split_stats(fold_val,   f"loso_{fold_name}_val"),
                "test":  split_stats(fold_test,  f"loso_{fold_name}_test"),
            }

        # -----------------------------------------------------------------------
        # CLASS BALANCE ANALYSIS
        # WHY: We want to know if any sentence has too few clips (e.g., only 5
        # signers × 1 clip = 5 total). If a sentence has < 4 clips total, it
        # might not appear in some folds at all — the model can never learn it.
        # -----------------------------------------------------------------------
        log.info("\n[CLASS BALANCE]")
        from collections import Counter
        sentence_counts = Counter(r["sentence_slug"] for r in manifest)
        low_count_sentences = {s: c for s, c in sentence_counts.items() if c < 5}
        if low_count_sentences:
            log.warning(f"  {len(low_count_sentences)} sentences have < 5 total clips:")
            for s, c in sorted(low_count_sentences.items(), key=lambda x: x[1]):
                log.warning(f"    '{s}': {c} clips")
        else:
            log.info(f"  All {len(sentence_counts)} sentences have ≥ 5 clips ✓")

        # Label distribution check
        labels = [int(r["label_id"]) for r in manifest if r["label_id"] != "-1"]
        label_counts = Counter(labels)
        min_c = min(label_counts.values())
        max_c = max(label_counts.values())
        log.info(f"  Label distribution: min={min_c}, max={max_c}, classes={len(label_counts)}")

        # -----------------------------------------------------------------------
        # SAVE SUMMARY
        # -----------------------------------------------------------------------
        summary_path = os.path.join(config.LOGS_DIR, "split_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info(f"\n  Summary saved → {summary_path}")

        log.info("\n" + "=" * 70)
        log.info("SPLITS SUMMARY")
        log.info("=" * 70)
        log.info(f"  Standard split: {len(std_train)} train / {len(std_val)} val / {len(std_test)} test")
        log.info(f"  LOSO folds:     7 folds × 3 CSVs each → {config.SPLITS_DIR}")
        log.info("\n✓ Splits complete. Ready for Step 05 (Export).")


    finally:
        log.removeHandler(fh)
        fh.close()

if __name__ == "__main__":
        run_splits()
