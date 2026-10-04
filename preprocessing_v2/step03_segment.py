"""
step03_segment.py — Temporal Segmentation (Uniform Sampling to T=32 frames)
=============================================================================

PURPOSE
-------
Convert every clip from its variable length (6–147 frames) to a fixed length
of T=32 frames using uniform temporal sampling.

THE VARIABLE-LENGTH PROBLEM
-----------------------------
Our frame counts per clip:
  Signer 1: avg 22.9,  min 7,   max 80
  Signer 2: avg 17.4,  min 6,   max 31
  Signer 3: avg 28.2,  min 15,  max 63
  ...
  Signer 7: avg 34.7,  min 13,  max 147

Neural networks (LSTMs, Transformers, 3D CNNs) operate on fixed-size tensors.
You cannot batch two sequences of length 7 and 147 together without padding —
and padding to 147 wastes 94% of the compute for short clips.

WHY T=32 SPECIFICALLY?
------------------------
  - The ISL literature (and the broader sign language recognition community)
    commonly uses 16, 32, or 64 frames.
  - T=16 is too short for multi-sign sentences (some signs alone take 15 frames).
  - T=64 is too long: clips as short as 6 frames would need 10× padding which
    causes the model to mostly learn "what padded frames look like."
  - T=32 fits comfortably in GPU memory as a batch of 32 clips = 32×32×543×3
    (≈ 1.7M float32 values = ~7MB per batch, very manageable).
  - Power of 2 is cache-friendly.

UNIFORM SAMPLING ALGORITHM
----------------------------
Given a clip of n_frames and target T:

Case 1: n_frames >= T (most clips)
  Select T indices evenly spaced across [0, n_frames-1].
  np.linspace(0, n_frames-1, T) gives T evenly-spaced floats; round to ints.
  
  Example: n=64, T=32 → indices [0, 2, 4, 6, ..., 62]
  This keeps the temporal dynamics proportional regardless of clip length.

Case 2: n_frames < T (short clips — mainly Signer 2)
  First, take all n_frames. Then repeat frames cyclically until we have T.
  Concretely: indices = [0, 1, ..., n-1, 0, 1, ..., n-1, ...] until T.
  
  WHY CYCLIC REPEAT AND NOT ZERO-PADDING?
  Zero-padding would insert T-n frames of [0,0,...,0] (shoulder midpoint in
  normalized space), which looks like the signer suddenly froze with both hands
  at the chest. The model would learn to detect that freeze pattern.
  Cyclic repeat makes the sign play in a loop — not physically accurate, but
  at least the hand positions are all plausible signing positions.

  WHY NOT NEAREST-NEIGHBOUR INTERPOLATION?
  For very short clips (n=6, T=32), each original frame would be repeated ~5
  times in a row, giving essentially the same result as cyclic. We use cyclic
  because it distributes the repetition more evenly.

OUTPUT FORMAT
-------------
One .npy file per clip in segments/:
  <sentence_slug>/<signer_id>/segment.npy  — shape (32, 543, 3)  float32

Plus a master manifest CSV listing every segment with:
  sentence, sentence_slug, signer_id, resolution, n_frames_original,
  n_frames_used, label_id, gloss
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

# ---------------------------------------------------------------------------
# LOGGING (Moved to run_segmentation to avoid side effects on import)
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# UNIFORM TEMPORAL SAMPLING
# ---------------------------------------------------------------------------

def uniform_sample(coords: np.ndarray, T: int) -> np.ndarray:
    """
    Uniformly sample T frames from a (n_frames, 543, 3) array.

    If n_frames >= T: subsample by evenly-spaced indices.
    If n_frames <  T: cyclic-repeat until we have T frames, then sample.

    Returns: (T, 543, 3) float32
    """
    n = coords.shape[0]

    if n == 0:
        raise ValueError("Cannot sample an empty clip (0 frames).")

    if n < T:
        # Cyclic repeat: tile the array until length >= T, then truncate to exactly T
        repeats_needed = (T + n - 1) // n   # ceiling division
        coords_tiled = np.tile(coords, (repeats_needed, 1, 1))
        return coords_tiled[:T].astype(np.float32)

    # Uniform subsampling: T evenly-spaced indices over [0, n-1]
    indices = np.round(np.linspace(0, n - 1, T)).astype(int)
    sampled = coords[indices]   # (T, 543, 3)

    return sampled.astype(np.float32)


# ---------------------------------------------------------------------------
# BUILD LABEL MAP
# Assign a unique integer ID to each sentence (for classification tasks).
# We sort alphabetically so the mapping is deterministic and reproducible.
# ---------------------------------------------------------------------------

def build_label_map(norm_base: Path) -> dict:
    """Return {sentence_slug: label_id} sorted alphabetically."""
    slugs = sorted([d.name for d in norm_base.iterdir() if d.is_dir()])
    return {slug: i for i, slug in enumerate(slugs)}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_segmentation():
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.SEGMENTS_DIR, exist_ok=True)
    fh = logging.FileHandler(os.path.join(config.LOGS_DIR, "step03_segment.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)
    try:
            
        log.info("=" * 70)
        log.info("ISL-CSLRT PREPROCESSING  —  Step 03: Temporal Segmentation")
        log.info(f"  Target frames T = {config.TARGET_FRAMES}")
        log.info("=" * 70)

        norm_base = Path(config.NORMALIZED_DIR)
        if not norm_base.exists():
            log.error(f"Normalized keypoints not found: {norm_base}")
            log.error("Run step02_normalize.py first.")
            sys.exit(1)

        # Load gloss map
        gloss_path = os.path.join(config.LOGS_DIR, "gloss_map.json")
        if os.path.exists(gloss_path):
            with open(gloss_path, encoding="utf-8") as f:
                gloss_map = json.load(f)
        else:
            gloss_map = {}
            log.warning("Gloss map not found — gloss column will be empty.")

        # Load the original clip index for frame counts
        index_path = os.path.join(config.LOGS_DIR, "clip_index.csv")
        original_counts = {}   # (sentence_slug, signer_id) → n_frames_original
        original_sentences = {} # (sentence_slug, signer_id) → sentence_original
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    slug = row["sentence"].replace(" ", "_").replace("/", "-").replace(",", "")
                    original_counts[(slug, row["signer_id"])] = int(row["n_frames"])
                    original_sentences[(slug, row["signer_id"])] = row["sentence"]

        label_map = build_label_map(norm_base)
        log.info(f"  Label classes: {len(label_map)} unique sentences")

        # Save label map for downstream use
        label_map_path = os.path.join(config.LOGS_DIR, "label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)
        log.info(f"  Label map saved → {label_map_path}")

        # Process every clip
        manifest_rows = []
        padded_clips  = 0
        sampled_clips = 0

        tasks = [
            (sent_dir.name, sig_dir.name)
            for sent_dir in sorted(norm_base.iterdir()) if sent_dir.is_dir()
            for sig_dir  in sorted(sent_dir.iterdir())  if sig_dir.is_dir()
        ]

        for sent_slug, sid in tqdm(tasks, desc="Segmenting", unit="clip"):
            in_path  = norm_base  / sent_slug / sid / "coords.npy"
            out_dir  = Path(config.SEGMENTS_DIR) / sent_slug / sid
            out_path = out_dir / "segment.npy"

            if not in_path.exists():
                log.warning(f"  Missing normalized coords: {in_path}")
                continue

            coords = np.load(str(in_path))    # (n_frames, 543, 3)
            n_orig = coords.shape[0]

            if n_orig < config.TARGET_FRAMES:
                padded_clips += 1

            try:
                segment = uniform_sample(coords, config.TARGET_FRAMES)  # (T, 543, 3)
            except ValueError as e:
                log.warning(f"  Skipping empty clip {sent_slug}/{sid}: {e}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), segment)

            # Determine resolution group from signer_id
            signer_int = int(sid)
            resolution = "low_res" if signer_int in config.LOW_RES_SIGNERS else "hd"

            # Reverse-map slug → original sentence name
            sentence_orig = original_sentences.get((sent_slug, sid), sent_slug.replace("_", " "))

            # Look up gloss (try original name and slug, lowercased to fix case mismatches)
            gloss = gloss_map.get(sentence_orig.lower(), gloss_map.get(sent_slug.lower(), ""))

            label_id = label_map.get(sent_slug, -1)
            pad_method = "cyclic" if n_orig < config.TARGET_FRAMES else "subsample"

            manifest_rows.append({
                "sentence":           sentence_orig,
                "sentence_slug":      sent_slug,
                "signer_id":          sid,
                "resolution":         resolution,
                "n_frames_original":  n_orig,
                "n_frames_sampled":   config.TARGET_FRAMES,
                "pad_method":         pad_method,
                "label_id":           label_id,
                "gloss":              gloss,
                "segment_path":       str(out_path),
            })
            sampled_clips += 1

        # Save manifest
        manifest_path = os.path.join(config.LOGS_DIR, "segment_manifest.csv")
        manifest_fields = ["sentence", "sentence_slug", "signer_id", "resolution",
                           "n_frames_original", "n_frames_sampled", "pad_method",
                           "label_id", "gloss", "segment_path"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerows(manifest_rows)

        log.info("\n" + "=" * 70)
        log.info("SEGMENTATION SUMMARY")
        log.info("=" * 70)
        log.info(f"  Total clips segmented : {sampled_clips}")
        log.info(f"  Clips cyclic-padded   : {padded_clips}  (had < {config.TARGET_FRAMES} frames)")
        log.info(f"  Manifest saved        : {manifest_path}")
        log.info(f"  Output segments       : {config.SEGMENTS_DIR}")
        log.info("\n✓ Segmentation complete. Ready for Step 04 (Splits).")


    finally:
        log.removeHandler(fh)
        fh.close()

if __name__ == "__main__":
        run_segmentation()
