"""
pipeline_v2/step02_normalize.py

Step 02 — Shoulder-anchor normalization + linear interpolation for tracking gaps.

Reads  : preprocessed_v5/keypoints_raw/<slug>/<signer>/coords.npy
                                                        confidence.npy
Writes : preprocessed_v5/keypoints_normalized/<slug>/<signer>/coords.npy
                                                               confidence.npy
                                                               gap_stats.json

Normalization
─────────────
We use the left-shoulder (pose landmark 11) and right-shoulder (pose landmark 12)
as reference anchors:
    origin  → midpoint of both shoulders
    scale   → Euclidean distance between shoulders (shoulder width)
All 543 landmark (x,y,z) coordinates are shifted and divided by this scale factor,
making the representation invariant to signer distance from the camera and body size.

Frames where the pose was not detected (zero coordinates) are linearly interpolated
from their neighbours and flagged in gap_stats.json.
"""

import os
import sys
import json
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(__file__))
import config

# Pose landmark indices for shoulder anchors (within the full 543 array)
L_SHOULDER = 11
R_SHOULDER = 12


def interpolate_gaps(coords: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Detect frames where pose was not detected (confidence[:,0] == 0)
    and linearly interpolate them from neighbouring valid frames.

    Returns the filled coords array and a list of gap records.
    """
    n_frames = coords.shape[0]
    # A frame is "missing" if pose shoulder landmarks are both zero
    missing = (confidence[:, L_SHOULDER] == 0.0) & (confidence[:, R_SHOULDER] == 0.0)

    filled = coords.copy()
    gap_records = []

    i = 0
    while i < n_frames:
        if missing[i]:
            gap_start = i
            while i < n_frames and missing[i]:
                i += 1
            gap_end = i  # exclusive

            length = gap_end - gap_start
            gap_records.append({"start": gap_start, "end": gap_end - 1, "length": length})

            # Find boundary frames
            before = gap_start - 1
            after  = gap_end

            if before < 0 and after >= n_frames:
                # Entire clip is missing — leave as zeros
                pass
            elif before < 0:
                # Fill from the right (forward fill backwards)
                filled[:gap_end] = coords[after]
            elif after >= n_frames:
                # Fill from the left (backward fill forwards)
                filled[gap_start:] = coords[before]
            else:
                # Linear interpolation between before and after frames
                for fi in range(gap_start, gap_end):
                    alpha = (fi - gap_start + 1) / (length + 1)
                    filled[fi] = (1 - alpha) * coords[before] + alpha * coords[after]
        else:
            i += 1

    return filled, gap_records


def normalize_clip(coords: np.ndarray) -> np.ndarray:
    """
    Shoulder-anchor normalization for each frame independently.
    Frames with zero shoulder coords (after interpolation) are left as-is.
    """
    normalized = coords.copy()
    n_frames   = coords.shape[0]

    for fi in range(n_frames):
        l_sh = coords[fi, L_SHOULDER]  # (3,)
        r_sh = coords[fi, R_SHOULDER]  # (3,)

        shoulder_width = np.linalg.norm(r_sh - l_sh)
        if shoulder_width < 1e-6:
            continue  # Cannot normalize; skip this frame

        origin = (l_sh + r_sh) / 2.0
        normalized[fi] = (coords[fi] - origin) / shoulder_width

    return normalized


def process_all(skip_existing: bool = True):
    if not os.path.isdir(config.KP_RAW_DIR):
        print(f"ERROR: keypoints_raw not found at {config.KP_RAW_DIR}")
        print("  → Run step01_extract_from_video.py first.")
        return

    processed = failed = 0

    for sent_slug in sorted(os.listdir(config.KP_RAW_DIR)):
        sent_raw_dir  = os.path.join(config.KP_RAW_DIR,  sent_slug)
        sent_norm_dir = os.path.join(config.KP_NORM_DIR, sent_slug)

        if not os.path.isdir(sent_raw_dir):
            continue

        for signer_id in sorted(os.listdir(sent_raw_dir)):
            raw_dir  = os.path.join(sent_raw_dir,  signer_id)
            norm_dir = os.path.join(sent_norm_dir, signer_id)

            coords_in  = os.path.join(raw_dir,  "coords.npy")
            coords_out = os.path.join(norm_dir, "coords.npy")

            if skip_existing and os.path.exists(coords_out):
                processed += 1
                continue

            if not os.path.exists(coords_in):
                continue

            try:
                coords     = np.load(coords_in)
                confidence = np.load(os.path.join(raw_dir, "confidence.npy"))

                # 1. Interpolate gaps
                filled, gap_records = interpolate_gaps(coords, confidence)

                # 2. Shoulder-anchor normalize
                normed = normalize_clip(filled)

                # 3. Save
                os.makedirs(norm_dir, exist_ok=True)
                np.save(coords_out,                               normed)
                np.save(os.path.join(norm_dir, "confidence.npy"), confidence)

                gap_summary = {
                    "n_frames":       coords.shape[0],
                    "n_gaps":         len(gap_records),
                    "total_missing":  sum(g["length"] for g in gap_records),
                    "max_gap":        max((g["length"] for g in gap_records), default=0),
                    "has_long_gap":   any(g["length"] > 8 for g in gap_records),
                    "gaps":           gap_records,
                }
                with open(os.path.join(norm_dir, "gap_stats.json"), "w") as f:
                    json.dump(gap_summary, f, indent=2)

                long = "⚠ long gap" if gap_summary["has_long_gap"] else ""
                print(f"  [OK] {sent_slug}/{signer_id}  "
                      f"gaps={gap_summary['n_gaps']}  "
                      f"max={gap_summary['max_gap']}  {long}")
                processed += 1

            except Exception as e:
                print(f"  [ERROR] {sent_slug}/{signer_id}: {e}")
                failed += 1

    print(f"\nStep 02 complete. Processed={processed}  Failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 02 — Normalize keypoints")
    parser.add_argument("--no_skip", action="store_true", help="Re-process existing outputs")
    args = parser.parse_args()
    process_all(skip_existing=not args.no_skip)

