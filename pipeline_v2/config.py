"""
pipeline_v2/config.py
Centralised configuration for the ISL-CSLRT v2 pipeline.
All paths are relative to the project root (ISL-CSLRT-Pipeline/).
"""

import os

# ─── Root Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CORPUS_ROOT  = os.path.join(PROJECT_ROOT, "ISL_CSLRT_Corpus")
VIDEOS_DIR   = os.path.join(CORPUS_ROOT,  "Videos_Sentence_Level")
FRAMES_DIR   = os.path.join(CORPUS_ROOT,  "Frames_Sentence_Level")   # legacy frames (backup)

OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "preprocessed_v5")         # fresh v2-pipeline outputs

KP_RAW_DIR   = os.path.join(OUTPUT_ROOT, "keypoints_raw")            # Step 01 output
KP_NORM_DIR  = os.path.join(OUTPUT_ROOT, "keypoints_normalized")     # Step 02 output
SEG_DIR      = os.path.join(OUTPUT_ROOT, "segments")                 # Step 03 output
SPLITS_DIR   = os.path.join(OUTPUT_ROOT, "splits")                   # Step 04 output
EXPORTS_DIR  = os.path.join(OUTPUT_ROOT, "exports")                  # Step 05 HDF5 output
PLOTS_DIR    = os.path.join(OUTPUT_ROOT, "plots")                    # QA plots
LOGS_DIR     = os.path.join(OUTPUT_ROOT, "logs")                     # CSVs and manifests
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "pipeline_v2", "reports") # keyframe reports

# ─── MediaPipe Landmark Layout (543 total) ─────────────────────────────────────
N_POSE       = 33   # indices   0 –  32
N_LHAND      = 21   # indices  33 –  53
N_RHAND      = 21   # indices  54 –  74
N_FACE       = 468  # indices  75 – 542
N_LANDMARKS  = N_POSE + N_LHAND + N_RHAND + N_FACE   # 543

# ─── 126-Landmark Subset used by models (same as training/dataset.py) ──────────
POSE_INDICES  = list(range(33))
LHAND_INDICES = list(range(33, 54))
RHAND_INDICES = list(range(54, 75))

# Key face landmarks: lips + eyebrows (relative to face block start at index 75)
_LIP_OFFSETS  = [0,13,14,17,37,39,40,61,78,80,81,82,84,87,88,91,95,146,178,181,191,
                 267,269,270,291,308,310,311,312,314,317,318,321,324,375,402,405,409,415]
_BROW_OFFSETS = [46,52,53,55,65,70,276,282,283,285,295,300]
FACE_INDICES  = [75 + o for o in _LIP_OFFSETS + _BROW_OFFSETS]  # 51 landmarks

SELECTED_INDICES = POSE_INDICES + LHAND_INDICES + RHAND_INDICES + FACE_INDICES  # 126 total
INPUT_SIZE       = len(SELECTED_INDICES) * 3                                     # 378 features

# ─── Temporal Sampling ─────────────────────────────────────────────────────────
T_FRAMES = 32          # fixed number of frames per segment

# ─── Dataset Splits (signer IDs) ───────────────────────────────────────────────
# Signers 1-2: mobile (low-res) | Signers 3-7: HD studio
TRAIN_SIGNERS = [3, 4, 5, 6]
VAL_SIGNERS   = [7]
TEST_SIGNERS  = [1, 2]

# ─── Training Defaults ─────────────────────────────────────────────────────────
HIDDEN_SIZE    = 256
DROPOUT        = 0.5
NUM_GRU_LAYERS = 2
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
MIN_LR         = 1e-5
DEFAULT_EPOCHS = 100

# ─── Model 3: LLM ──────────────────────────────────────────────────────────────
OLLAMA_MODEL   = "gemma2:2b"   # Ollama model tag for Gemma 2B
OLLAMA_URL     = "http://localhost:11434/api/generate"

