"""
step02_normalize.py — Keypoint Normalization + Interpolation
=============================================================

PURPOSE
-------
Take the raw (x, y, z) coordinates from Step 01 and transform them so the
model learns sign SHAPES and MOTIONS, not signer-specific positions.

WHY RAW COORDINATES ARE PROBLEMATIC
------------------------------------
MediaPipe returns coordinates in image-space [0, 1]:
  - x=0 → left edge of image, x=1 → right edge
  - y=0 → top edge, y=1 → bottom edge

This means:
  1. If Signer 3 stands 1m from the camera and Signer 5 stands 2m away,
     their wrist positions will be at completely different (x, y) values
     even if they are performing the same sign with the same hand shape.
     → The model would learn "Signer 3 has large arm movements" instead of
       "this is the sign for HAPPY."

  2. If Signer 2 is filmed portrait (240×320) and Signer 6 is filmed landscape
     (1920×1080), the aspect ratios differ — so the same gesture traces a
     different path through coordinate space.
     → The model would learn image orientation, not sign.

  3. Some signers are taller, some shorter — their shoulder height in the
     image differs even if they stand at the same distance.

WHAT NORMALIZATION FIXES
-------------------------
Step A — Translate to shoulder midpoint:
  We compute the midpoint between left shoulder (landmark 11) and right
  shoulder (landmark 12). We subtract this from every landmark.
  Result: the origin (0,0,0) is now the signer's shoulder centre.
  → Fixes problem 1 (translation invariance).

Step B — Scale by shoulder width:
  We compute the Euclidean distance between both shoulders (the "shoulder
  width"). We divide every coordinate by this distance.
  Result: the shoulder width is always exactly 1.0.
  → Fixes problems 2 and 3 (scale invariance, body size invariance).

WHY SHOULDER LANDMARKS AS ANCHOR?
-----------------------------------
  - The shoulders are the most consistently detected Pose landmarks — they
    are almost never occluded in frontal signing videos.
  - They define a stable, signer-relative coordinate frame that maps directly
    to the signing space ISL uses (signs are typically performed within arm's
    reach of the shoulder plane).
  - The alternative (nose tip) is unstable when the head turns; hip midpoint
    is too far from the signing space.

Step C — Interpolate missing detections:
  When confidence[frame, landmark_group] == 0, it means MediaPipe did not
  detect that landmark group in that frame (e.g. one hand moved off-screen).
  We use linear interpolation across time to fill those gaps.
  If the group is missing at the start or end, we forward-fill or back-fill.

  WHY INTERPOLATE RATHER THAN ZERO-FILL?
  A hand at position (0,0,0) after normalization means "hand is at the
  shoulder midpoint." That is a meaningful position — a sign could pass
  through it. Filling with zeros would inject false sign information.
  Linear interpolation assumes the hand moved smoothly between its last
  known position and its next known position — usually correct for signing.

OUTPUT FORMAT
-------------
Same structure as Step 01, but in keypoints_normalized/:
  coords.npy      — (n_frames, 543, 3) float32  — normalized coordinates
  confidence.npy  — (n_frames, 543)   float32  — original confidence (unchanged)
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
os.makedirs(config.LOGS_DIR, exist_ok=True)
os.makedirs(config.NORMALIZED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "step02_normalize.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LANDMARK GROUP SLICES
# These define which indices in the 543-landmark array belong to each group.
# We interpolate per group — either the whole hand is detected or it isn't.
# ---------------------------------------------------------------------------

POSE_SLICE       = slice(0,   33)
LEFT_HAND_SLICE  = slice(33,  54)
RIGHT_HAND_SLICE = slice(54,  75)
FACE_SLICE       = slice(75, 543)

LANDMARK_GROUPS = [
    ("pose",        POSE_SLICE),
    ("left_hand",   LEFT_HAND_SLICE),
    ("right_hand",  RIGHT_HAND_SLICE),
    ("face",        FACE_SLICE),
]


# ---------------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_clip(coords: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """
    Normalize a clip's coordinates using shoulder-anchor method.

    Parameters
    ----------
    coords     : (n_frames, 543, 3) raw MediaPipe coordinates
    confidence : (n_frames, 543)    detection confidence (0 = missing)

    Returns
    -------
    norm_coords : (n_frames, 543, 3) normalized coordinates
    """
    n_frames = coords.shape[0]
    norm = coords.copy()

    # Pose landmarks are at indices 0–32 in our layout.
    # Left shoulder  = index 11 within the pose group = global index 11
    # Right shoulder = index 12 within the pose group = global index 12

    L_SHOULDER = config.POSE_LEFT_SHOULDER_IDX   # 11
    R_SHOULDER = config.POSE_RIGHT_SHOULDER_IDX  # 12

    for f in range(n_frames):
        left_conf  = confidence[f, L_SHOULDER]
        right_conf = confidence[f, R_SHOULDER]

        # If BOTH shoulders are detected with confidence > 0.3:
        if left_conf > 0.3 and right_conf > 0.3:
            left_sh  = norm[f, L_SHOULDER]   # (3,)
            right_sh = norm[f, R_SHOULDER]   # (3,)

            # Step A: anchor = midpoint between shoulders
            anchor = (left_sh + right_sh) / 2.0   # (3,)

            # Step B: scale = Euclidean distance between shoulders
            shoulder_width = np.linalg.norm(left_sh - right_sh)

            if shoulder_width > 1e-6:   # avoid division by zero
                norm[f] = (norm[f] - anchor) / shoulder_width
            else:
                # Degenerate case: shoulders at the same point (rare)
                # Just subtract the anchor, skip scaling
                norm[f] = norm[f] - anchor
        # If shoulders are not detected, we leave this frame's coordinates
        # as-is for now. The interpolation step below will fill it in using
        # neighboring frames where normalization succeeded.

    return norm


# ---------------------------------------------------------------------------
# INTERPOLATION
# ---------------------------------------------------------------------------

def interpolate_missing(coords: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """
    Fill frames where a landmark group has zero confidence using linear
    interpolation across the time axis.

    WHY PER-GROUP:
      Hands go off-screen as a unit. We interpolate the entire hand (21 lms)
      together — either all 21 landmarks are missing or all are present.
      Interpolating each landmark independently could introduce inconsistent
      finger configurations that never actually occurred.

    Edge cases:
      - Missing at start → forward-fill from first valid frame
      - Missing at end   → back-fill from last valid frame
      - All missing      → leave as zeros (will be flagged in the index)
    """
    n_frames = coords.shape[0]
    filled = coords.copy()

    for group_name, slc in LANDMARK_GROUPS:
        group_conf = confidence[:, slc]          # (n_frames, group_size)

        # A frame has this group "present" if ANY landmark in the group
        # has confidence > 0.3
        present = (group_conf.max(axis=1) > 0.3)  # (n_frames,) bool

        # Nothing to do if all frames are present or all are missing
        if present.all() or not present.any():
            continue

        present_indices = np.where(present)[0]
        missing_indices = np.where(~present)[0]

        for f_miss in missing_indices:
            # Find the closest valid frame before and after
            before = present_indices[present_indices < f_miss]
            after  = present_indices[present_indices > f_miss]

            if len(before) == 0 and len(after) == 0:
                continue   # entire group missing — leave as zeros

            elif len(before) == 0:
                # Missing before any valid detection → forward-fill
                filled[f_miss, slc] = filled[after[0], slc]

            elif len(after) == 0:
                # Missing after all valid detections → back-fill
                filled[f_miss, slc] = filled[before[-1], slc]

            else:
                # Linear interpolation between nearest before and after
                f_before = before[-1]
                f_after  = after[0]
                t = (f_miss - f_before) / (f_after - f_before)   # in [0, 1]
                filled[f_miss, slc] = (
                    (1 - t) * filled[f_before, slc] +
                    t        * filled[f_after,  slc]
                )

    return filled


# ---------------------------------------------------------------------------
# PROCESS ONE CLIP
# ---------------------------------------------------------------------------

def process_clip(sentence_slug: str, signer_id: str) -> bool:
    """Load raw keypoints, normalize, interpolate, save."""

    in_dir  = Path(config.KEYPOINTS_DIR)  / sentence_slug / signer_id
    out_dir = Path(config.NORMALIZED_DIR) / sentence_slug / signer_id

    coords_path = in_dir / "coords.npy"
    conf_path   = in_dir / "confidence.npy"

    if not coords_path.exists():
        log.warning(f"  Missing: {coords_path}")
        return False

    coords     = np.load(str(coords_path))       # (n_frames, 543, 3)
    confidence = np.load(str(conf_path))         # (n_frames, 543)

    if coords.shape[0] == 0:
        log.warning(f"  Empty clip: {sentence_slug}/{signer_id}")
        return False

    # --- Normalize ---
    norm = normalize_clip(coords, confidence)

    # --- Interpolate ---
    norm = interpolate_missing(norm, confidence)

    # --- Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "coords.npy"),     norm)
    np.save(str(out_dir / "confidence.npy"), confidence)   # pass through unchanged

    return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_normalization():
    log.info("=" * 70)
    log.info("ISL-CSLRT PREPROCESSING  —  Step 02: Normalization + Interpolation")
    log.info("=" * 70)

    kp_base = Path(config.KEYPOINTS_DIR)
    if not kp_base.exists():
        log.error(f"Keypoints directory not found: {kp_base}")
        log.error("Run step01_extract_keypoints.py first.")
        sys.exit(1)

    # Collect all (sentence_slug, signer_id) pairs from the output of Step 01
    tasks = []
    for sent_dir in sorted(kp_base.iterdir()):
        if not sent_dir.is_dir():
            continue
        for signer_dir in sorted(sent_dir.iterdir()):
            if not signer_dir.is_dir():
                continue
            tasks.append((sent_dir.name, signer_dir.name))

    log.info(f"Found {len(tasks)} clips to normalize")

    success = 0
    skip    = 0

    for sent_slug, sid in tqdm(tasks, desc="Normalizing", unit="clip"):
        ok = process_clip(sent_slug, sid)
        if ok: success += 1
        else:  skip    += 1

    log.info("\n" + "=" * 70)
    log.info("NORMALIZATION SUMMARY")
    log.info("=" * 70)
    log.info(f"  Clips normalized : {success}")
    log.info(f"  Clips skipped    : {skip}")
    log.info(f"  Output           : {config.NORMALIZED_DIR}")
    log.info("\n✓ Normalization complete. Ready for Step 03 (Segmentation).")


if __name__ == "__main__":
    run_normalization()
