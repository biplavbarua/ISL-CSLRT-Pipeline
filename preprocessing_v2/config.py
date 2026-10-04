"""
config.py — Central Configuration for ISL-CSLRT Preprocessing Pipeline
=======================================================================

WHY A SINGLE CONFIG FILE?
--------------------------
All preprocessing scripts agree on the same paths and parameters.
Change any value here once — every script picks it up automatically.
This is called "single source of truth."
"""

import os

# ---------------------------------------------------------------------------
# ROOT PATHS
# Directory layout:
#   <workspace>/                         ← WORKSPACE_ROOT
#     ISL_CSLRT_Corpus/                  ← CORPUS_ROOT  (the data)
#       Frames_Sentence_Level/
#       Frames_Word_Level/
#       Videos_Sentence_Level/
#       corpus_csv_files/
#     preprocessing/                     ← this file lives here
#     preprocessed/                      ← OUTPUT_ROOT  (we write here)
# ---------------------------------------------------------------------------

# preprocessing/ directory → go up one level → workspace root
_PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT    = os.environ.get("ISL_WORKSPACE_ROOT", os.path.dirname(_PREPROCESSING_DIR))

CORPUS_ROOT = os.path.join(_WORKSPACE_ROOT, "ISL_CSLRT_Corpus")
OUTPUT_ROOT = os.path.join(_WORKSPACE_ROOT, "preprocessed_v4")

FRAMES_SENTENCE_DIR = os.path.join(CORPUS_ROOT, "Frames_Sentence_Level")
FRAMES_WORD_DIR     = os.path.join(CORPUS_ROOT, "Frames_Word_Level")
VIDEOS_SENTENCE_DIR = os.path.join(CORPUS_ROOT, "Videos_Sentence_Level")
GLOSS_CSV           = os.path.join(CORPUS_ROOT, "corpus_csv_files", "ISL Corpus sign glosses.csv")

KEYPOINTS_DIR  = os.path.join(OUTPUT_ROOT, "keypoints_raw")
NORMALIZED_DIR = os.path.join(OUTPUT_ROOT, "keypoints_normalized")
SEGMENTS_DIR   = os.path.join(OUTPUT_ROOT, "segments")
SPLITS_DIR     = os.path.join(OUTPUT_ROOT, "splits")
LOGS_DIR       = os.path.join(OUTPUT_ROOT, "logs")

assert KEYPOINTS_DIR.startswith(OUTPUT_ROOT), "KEYPOINTS_DIR must resolve under OUTPUT_ROOT (preprocessed_v4/)"
assert NORMALIZED_DIR.startswith(OUTPUT_ROOT), "NORMALIZED_DIR must resolve under OUTPUT_ROOT (preprocessed_v4/)"

# ---------------------------------------------------------------------------
# SIGNER METADATA
# Signers 1-2: mobile phone  → 240x320 portrait
# Signers 3-7: DSLR camera   → 1920x1080 landscape
# ---------------------------------------------------------------------------

ALL_SIGNERS     = [1, 2, 3, 4, 5, 6, 7]
LOW_RES_SIGNERS = [1, 2]
HD_SIGNERS      = [3, 4, 5, 6, 7]

# ---------------------------------------------------------------------------
# PIPELINE CONFIGURATION
# ---------------------------------------------------------------------------
# List of clips to explicitly exclude from dataset (e.g. corrupted, zero-content)
# Format: "sentence_slug/signer_id"
EXCLUDED_CLIPS = {
    "he_is_on_the_way/3",  # Zero hand detections (hands completely out of frame)
}

# ---------------------------------------------------------------------------
# MEDIAPIPE CONFIGURATION (Legacy / Unused)
# Note: The parameters below match the old mp.solutions.holistic API.
# They are no longer imported by step01_extract_keypoints.py, which now
# uses the Tasks API directly.
# ---------------------------------------------------------------------------

MEDIAPIPE_CONFIG = {
    "static_image_mode":        True,
    "model_complexity":         1,
    "enable_segmentation":      False,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence":  0.5,
}

# MediaPipe Holistic landmarks:
#   Pose: 33  |  Left Hand: 21  |  Right Hand: 21  |  Face: 468  |  TOTAL: 543
N_POSE_LANDMARKS      = 33
N_HAND_LANDMARKS      = 21
N_FACE_LANDMARKS      = 468
N_TOTAL_LANDMARKS     = N_POSE_LANDMARKS + N_HAND_LANDMARKS * 2 + N_FACE_LANDMARKS  # 543
N_COORDS_PER_LANDMARK = 3  # x, y, z

# ---------------------------------------------------------------------------
# NORMALIZATION
# Anchor: midpoint between left shoulder (11) and right shoulder (12).
# Scale: Euclidean distance between the two shoulders.
# Result: translation-invariant and scale-invariant landmark coordinates.
# ---------------------------------------------------------------------------

POSE_LEFT_SHOULDER_IDX  = 11
POSE_RIGHT_SHOULDER_IDX = 12

# ---------------------------------------------------------------------------
# TEMPORAL SEGMENTATION
# T=32: standard in SLR literature; fits clips of 6-147 frames without extreme
# padding; power of 2 for GPU efficiency.
# ---------------------------------------------------------------------------

TARGET_FRAMES = 32

# ---------------------------------------------------------------------------
# SPLIT CONFIGURATION
# LOSO = Leave-One-Signer-Out: each fold holds one signer out for test.
# Standard split: train on HD signers, test on low-res (hardest challenge).
# ---------------------------------------------------------------------------

ALL_SIGNERS_LIST = list(range(1, 8))

LOSO_FOLDS = {
    "fold_1": {"test": 1, "val": 2, "train": [3, 4, 5, 6, 7]},
    "fold_2": {"test": 2, "val": 1, "train": [3, 4, 5, 6, 7]},
    "fold_3": {"test": 3, "val": 4, "train": [1, 2, 5, 6, 7]},
    "fold_4": {"test": 4, "val": 5, "train": [1, 2, 3, 6, 7]},
    "fold_5": {"test": 5, "val": 6, "train": [1, 2, 3, 4, 7]},
    "fold_6": {"test": 6, "val": 7, "train": [1, 2, 3, 4, 5]},
    "fold_7": {"test": 7, "val": 6, "train": [1, 2, 3, 4, 5]}
}

STANDARD_SPLIT = {
    "train": [3, 4, 5, 6],
    "val":   [7],
    "test":  [1, 2],
}

# ---------------------------------------------------------------------------
# MISC
# ---------------------------------------------------------------------------

RANDOM_SEED      = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SKIP_FILENAMES   = {"Thumbs.db", ".DS_Store", "desktop.ini"}
