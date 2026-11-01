"""
pipeline_v2/step03_segment.py

Step 03 — Uniform temporal sampling to produce fixed-length T=32 segments.

Reads  : preprocessed_v5/keypoints_normalized/<slug>/<signer>/coords.npy
Writes : preprocessed_v5/segments/<slug>/<signer>/segment.npy  — (T, 543, 3) float32
"""

import os
import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(__file__))
import config


def uniform_sample(coords: np.ndarray, t: int = config.T_FRAMES) -> np.ndarray:
    """
    Sample exactly `t` frames from `coords` using uniform spacing.
    - If n_frames >= t: select t evenly-spaced frames (linspace indices).
    - If n_frames < t:  repeat the sequence cyclically to reach t frames,
                        then uniform sample.
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros((t, config.N_LANDMARKS, 3), dtype=np.float32)

    if n < t:
        # Tile the clip until it's long enough
        repeats = (t // n) + 1
        coords  = np.tile(coords, (repeats, 1, 1))
        n       = coords.shape[0]

    indices = np.linspace(0, n - 1, t, dtype=int)
    return coords[indices].astype(np.float32)


def process_all(skip_existing: bool = True):
    if not os.path.isdir(config.KP_NORM_DIR):
        print(f"ERROR: keypoints_normalized not found at {config.KP_NORM_DIR}")
        print("  → Run step02_normalize.py first.")
        return

    processed = failed = 0

    for sent_slug in sorted(os.listdir(config.KP_NORM_DIR)):
        sent_norm_dir = os.path.join(config.KP_NORM_DIR, sent_slug)
        sent_seg_dir  = os.path.join(config.SEG_DIR,     sent_slug)

        if not os.path.isdir(sent_norm_dir):
            continue

        for signer_id in sorted(os.listdir(sent_norm_dir)):
            norm_dir  = os.path.join(sent_norm_dir, signer_id)
            seg_dir   = os.path.join(sent_seg_dir,  signer_id)
            seg_path  = os.path.join(seg_dir, "segment.npy")
            coords_in = os.path.join(norm_dir, "coords.npy")

            if skip_existing and os.path.exists(seg_path):
                processed += 1
                continue

            if not os.path.exists(coords_in):
                continue

            try:
                coords  = np.load(coords_in)     # (n_frames, 543, 3)
                segment = uniform_sample(coords)  # (T, 543, 3)

                os.makedirs(seg_dir, exist_ok=True)
                np.save(seg_path, segment)
                print(f"  [OK] {sent_slug}/{signer_id}  "
                      f"src_frames={coords.shape[0]} → {config.T_FRAMES} sampled")
                processed += 1

            except Exception as e:
                print(f"  [ERROR] {sent_slug}/{signer_id}: {e}")
                failed += 1

    print(f"\nStep 03 complete. Processed={processed}  Failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 03 — Temporal sampling")
    parser.add_argument("--no_skip", action="store_true")
    args = parser.parse_args()
    process_all(skip_existing=not args.no_skip)

