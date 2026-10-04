"""
step01_extract_keypoints.py — Keypoint Extraction (Three-Landmarker Approach)
==============================================================================

WHY THREE SEPARATE LANDMARKERS INSTEAD OF HOLISTIC?
-----------------------------------------------------
MediaPipe's "Holistic" solution (mp.solutions.holistic) was only available
on Python ≤ 3.11. Python 3.13 (your version) requires the newer Tasks API
(mediapipe 0.10.x), which provides three separate, independent landmarkers:
  - PoseLandmarker   → 33 body landmarks
  - HandLandmarker   → up to 2 hands × 21 landmarks, labelled LEFT/RIGHT
  - FaceLandmarker   → 478 landmarks (we use first 468 to match Holistic)

We run all three on each frame and assemble the output into the same
543-landmark array, preserving full compatibility with Steps 2–5.

WHY THREE SEPARATE MODELS NOT JUST POSE+HAND?
----------------------------------------------
Face landmarks capture non-manual features (eyebrows, lip shape, gaze)
that are linguistically meaningful in ISL for marking questions, negation,
and intensity. Including them now costs nothing extra and gives future
experiments the option to use them without re-running extraction.

OUTPUT FORMAT (identical to original design)
--------------------------------------------
  coords.npy      (n_frames, 543, 3) float32 — (x,y,z) per landmark
  confidence.npy  (n_frames, 543)    float32 — 0.0=missing, ~1.0=detected

Landmark index layout:
  [0 :33)  — Pose        (33 landmarks)
  [33:54)  — Left Hand   (21 landmarks)
  [54:75)  — Right Hand  (21 landmarks)
  [75:543) — Face        (468 landmarks)

MODEL FILES (auto-downloaded on first run):
  preprocessed/models/pose_landmarker_full.task
  preprocessed/models/hand_landmarker.task
  preprocessed/models/face_landmarker.task
"""

import os
import sys
import csv
import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
os.makedirs(config.LOGS_DIR,      exist_ok=True)
os.makedirs(config.KEYPOINTS_DIR, exist_ok=True)
MODELS_DIR = os.path.join(config.OUTPUT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "step01_extract.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)

# Model download URLs and local paths
MODELS = {
    "pose": {
        "url":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "path": os.path.join(MODELS_DIR, "pose_landmarker_full.task"),
    },
    "hand": {
        "url":  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "path": os.path.join(MODELS_DIR, "hand_landmarker.task"),
    },
    "face": {
        "url":  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "path": os.path.join(MODELS_DIR, "face_landmarker.task"),
    },
}

N_FACE_USE = 468   # FaceLandmarker returns 478; we use first 468 to match the
                   # original Holistic layout and keep N_TOTAL_LANDMARKS=543


# ---------------------------------------------------------------------------
# MODEL DOWNLOAD
# ---------------------------------------------------------------------------

def ensure_models():
    """Download any missing model bundles."""
    for name, info in MODELS.items():
        path = info["path"]
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            log.info(f"  Model '{name}' already present ({os.path.getsize(path)//1024}KB)")
            continue
        log.info(f"  Downloading '{name}' model → {path}")
        urllib.request.urlretrieve(info["url"], path)
        log.info(f"  Done ({os.path.getsize(path)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# LANDMARKER INITIALISATION
# ---------------------------------------------------------------------------

def init_landmarkers():
    """
    Initialise three MediaPipe Tasks landmarkers for IMAGE (static) mode.

    WHY IMAGE MODE:
      IMAGE mode processes each frame independently with no temporal smoothing.
      Our Step 2 normalization does its own temporal consistency via interpolation.
      MediaPipe's built-in video smoothing would interfere with anchor normalization.

    WHY num_poses=1, num_hands=2:
      We always have exactly one signer per clip. Detecting multiple poses would
      waste time. We allow 2 hands because signing uses both simultaneously.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions,
        HandLandmarker, HandLandmarkerOptions,
        FaceLandmarker, FaceLandmarkerOptions,
        RunningMode,
    )

    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELS["pose"]["path"]),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELS["hand"]["path"]),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELS["face"]["path"]),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    pose_det = PoseLandmarker.create_from_options(pose_opts)
    hand_det = HandLandmarker.create_from_options(hand_opts)
    face_det = FaceLandmarker.create_from_options(face_opts)

    log.info("All three landmarkers initialised (Pose + Hand + Face)")
    return pose_det, hand_det, face_det, mp


# ---------------------------------------------------------------------------
# EXTRACT LANDMARKS FROM ONE FRAME
# ---------------------------------------------------------------------------

def extract_landmarks(pose_det, hand_det, face_det, mp, img_bgr: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all three landmarkers on one BGR frame.
    Returns:
      coords     (543, 3)  float32 — (x, y, z)
      confidence (543,)    float32 — 0.0 if not detected
    """
    N = config.N_TOTAL_LANDMARKS
    coords     = np.zeros((N, 3), dtype=np.float32)
    confidence = np.zeros(N,      dtype=np.float32)

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    idx = 0   # running landmark index into coords/confidence arrays

    # --- POSE (33 landmarks, index 0–32) ---
    pose_result = pose_det.detect(mp_image)
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            coords[idx]     = [lm.x, lm.y, lm.z]
            confidence[idx] = lm.visibility if hasattr(lm, 'visibility') else 1.0
            idx += 1
    else:
        idx += config.N_POSE_LANDMARKS   # leave as zeros

    # --- HANDS (left: 33–53, right: 54–74) ---
    # HandLandmarker returns up to 2 hands with handedness labels.
    # We must assign each detected hand to the correct slot (left or right).
    hand_result = hand_det.detect(mp_image)

    left_hand_lms  = None
    right_hand_lms = None

    if hand_result.hand_landmarks:
        for hand_lms, handedness_list in zip(hand_result.hand_landmarks,
                                              hand_result.handedness):
            # handedness_list[0].category_name is "Left" or "Right"
            # Note: MediaPipe labels from the CAMERA's perspective, not the
            # signer's — so "Right" in MediaPipe = signer's right hand.
            label = handedness_list[0].category_name
            if label == "Left":
                left_hand_lms = hand_lms
            elif label == "Right":
                right_hand_lms = hand_lms

    # Left hand slot
    if left_hand_lms:
        for lm in left_hand_lms:
            coords[idx]     = [lm.x, lm.y, lm.z]
            confidence[idx] = 1.0
            idx += 1
    else:
        idx += config.N_HAND_LANDMARKS

    # Right hand slot
    if right_hand_lms:
        for lm in right_hand_lms:
            coords[idx]     = [lm.x, lm.y, lm.z]
            confidence[idx] = 1.0
            idx += 1
    else:
        idx += config.N_HAND_LANDMARKS

    # --- FACE (468 landmarks, index 75–542) ---
    # FaceLandmarker returns 478 landmarks; we use first 468 to keep our
    # total at 543 (matching the original Holistic output format).
    face_result = face_det.detect(mp_image)
    if face_result.face_landmarks:
        for lm in face_result.face_landmarks[0][:N_FACE_USE]:
            coords[idx]     = [lm.x, lm.y, lm.z]
            confidence[idx] = 1.0
            idx += 1
    # (if face not detected, those 468 slots stay at zeros)

    return coords, confidence


# ---------------------------------------------------------------------------
# PROCESS ONE CLIP
# ---------------------------------------------------------------------------

def process_clip(pose_det, hand_det, face_det, mp, clip: dict, landmark_subset="full") -> bool:
    """
    Process all frames for one (sentence, signer) clip.
    Saves coords.npy and confidence.npy. Returns True on success.
    """
    sentence    = clip["sentence"]
    signer_id   = int(clip["signer_id"])
    frames_dir  = Path(clip["frames_dir"])
    frame_files = json.loads(clip["frame_files"])

    if not frame_files:
        return False

    n_frames       = len(frame_files)
    all_coords     = np.zeros((n_frames, config.N_TOTAL_LANDMARKS, config.N_COORDS_PER_LANDMARK), dtype=np.float32)
    all_confidence = np.zeros((n_frames, config.N_TOTAL_LANDMARKS), dtype=np.float32)

    for i, fname in enumerate(frame_files):
        img_path = frames_dir / fname
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        coords, conf = extract_landmarks(pose_det, hand_det, face_det, mp, img_bgr)
        all_coords[i]     = coords
        all_confidence[i] = conf

    # Save — use filesystem-safe sentence slug
    sentence_slug = sentence.replace(" ", "_").replace("/", "-").replace(",", "")
    out_dir = Path(config.KEYPOINTS_DIR) / sentence_slug / str(signer_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if landmark_subset == "pose_hands":
        all_coords = all_coords[:, :75, :]
        all_confidence = all_confidence[:, :75]

    np.save(str(out_dir / "coords.npy"),     all_coords)
    np.save(str(out_dir / "confidence.npy"), all_confidence)
    return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_extraction(limit=None, landmark_subset="full"):
    log.info("=" * 70)
    log.info("ISL-CSLRT PREPROCESSING  —  Step 01: Keypoint Extraction")
    log.info("  Strategy: Pose + Hand + Face landmarkers (mediapipe 0.10.x Tasks API)")
    log.info("=" * 70)

    # Ensure model bundles are present
    log.info("\n[MODELS] Checking model bundles...")
    ensure_models()

    # Load clip index from Step 00
    index_path = os.path.join(config.LOGS_DIR, "clip_index.csv")
    if not os.path.exists(index_path):
        log.error("Clip index not found. Run step00_clean.py first.")
        sys.exit(1)

    with open(index_path, encoding="utf-8") as f:
        clips = [r for r in csv.DictReader(f) if int(r["n_frames"]) > 0]
        if limit is not None:
            clips = clips[:limit]

    total_frames = sum(int(c["n_frames"]) for c in clips)
    log.info(f"\nProcessing {len(clips)} clips  ({total_frames:,} frames total)")
    log.info("Estimated time: 15–45 minutes on CPU.")

    pose_det, hand_det, face_det, mp_mod = init_landmarkers()

    success = 0
    skip    = 0
    t0      = time.time()

    for clip in tqdm(clips, desc="Extracting keypoints", unit="clip"):
        ok = process_clip(pose_det, hand_det, face_det, mp_mod, clip, landmark_subset=landmark_subset)
        if ok: success += 1
        else:  skip    += 1

    pose_det.close()
    hand_det.close()
    face_det.close()

    elapsed = time.time() - t0

    log.info("\n" + "=" * 70)
    log.info("EXTRACTION SUMMARY")
    log.info("=" * 70)
    log.info(f"  Clips processed  : {success}")
    log.info(f"  Clips skipped    : {skip}")
    log.info(f"  Frames processed : {total_frames:,}")
    log.info(f"  Time elapsed     : {elapsed/60:.1f} min")
    log.info(f"  Avg speed        : {total_frames/elapsed:.1f} frames/sec")
    log.info(f"\n  Output written to: {config.KEYPOINTS_DIR}")
    log.info("\n✓ Extraction complete. Ready for Step 02 (Normalization).")


if __name__ == "__main__":
    run_extraction()
