# ISL-CSLRT Preprocessing Pipeline

Five-step preprocessing pipeline that transforms the raw ISL-CSLRT corpus
into signer-independent, model-ready datasets for Sign Language Recognition research.

## Quick Start

```bash
# Option A: Rely on the default relative path
cd path/to/ISL_CSLRT_Corpus/preprocessing_v2

# Option B: Explicitly set the workspace root via environment variable
export ISL_WORKSPACE_ROOT=/path/to/ISL_CSLRT_Corpus
cd $ISL_WORKSPACE_ROOT/preprocessing_v2

# Install dependencies (once)
pip3 install -r requirements.txt

# Run the full pipeline (Steps 1 through 5)
python3 run_pipeline.py
```

## Pipeline Steps

| Script | Step | What It Does | Output |
|---|---|---|---|
| `step00_clean.py` | 0 | Audit dataset, build clip index, cross-check glosses | `logs/clip_index.csv` |
| `step01_extract_keypoints.py` | 1 | MediaPipe Holistic → 543 keypoints per frame | `keypoints_raw/<sent>/<signer>/` |
| `step02_normalize.py` | 2 | Shoulder-anchor normalisation + linear interpolation | `keypoints_normalized/<sent>/<signer>/` |
| `step03_segment.py` | 3 | Uniform temporal sampling → T=32 frames per clip | `segments/<sent>/<signer>/segment.npy` |
| `step04_build_splits.py` | 4 | Signer-independent LOSO + standard splits | `splits/*.csv` |
| `step05_export.py` | 5 | Pack splits into HDF5 + QA plots | `exports/*.h5`, `plots/*.png` |

## Output Structure

The current configuration reads legacy v1 keypoints from `preprocessed/` and writes all new downstream artifacts to `preprocessed_v4/`.

```
preprocessed/                   # Legacy v1 Keypoint Cache (Read-Only)
├── keypoints_raw/              # Step 01: raw MediaPipe coordinates
│   └── <sentence_slug>/
│       └── <signer_id>/
│           ├── coords.npy      # (n_frames, 543, 3) float32
│           └── confidence.npy  # (n_frames, 543) float32
└── keypoints_normalized/       # Step 02: shoulder-normalised + interpolated
    └── (same structure)

preprocessed_v4/                # Canonical v4 Output
├── segments/                   # Step 03: fixed T=32 frame clips
│   └── <sentence_slug>/
│       └── <signer_id>/
│           └── segment.npy    # (32, 543, 3) float32
├── splits/                    # Step 04: split CSV manifests
│   ├── standard_train.csv
│   ├── standard_val.csv
│   ├── standard_test.csv
│   ├── loso_fold_1_train.csv
│   └── ...
├── exports/                   # Step 05: packed HDF5 files
│   ├── standard_train.h5
│   └── ...
├── plots/                     # Step 05: QA visualisations
│   └── dataset_qa.png
└── logs/                      # Audit trails and manifests
    ├── clip_index.csv
    ├── gloss_map.json
    ├── label_map.json
    ├── segment_manifest.csv
    └── split_summary.json
```

## Landmark Layout (543 total)

| Index Range | Group | Count |
|---|---|---|
| 0 – 32 | Pose (body) | 33 |
| 33 – 53 | Left Hand | 21 |
| 54 – 74 | Right Hand | 21 |
| 75 – 542 | Face | 468 |

## Loading Preprocessed Data

```python
import h5py
import numpy as np

with h5py.File("preprocessed_v4/exports/standard_train.h5", "r") as hf:
    clips    = hf["clips"][:]      # (N, 32, 543, 3) float32
    labels   = hf["labels"][:]     # (N,) int32
    signers  = hf["signer_ids"][:] # (N,) int32
    glosses  = hf["glosses"][:]    # (N,) str

print(f"Training set: {clips.shape[0]} clips")
```

## Research Context

**Note on Provenance:** Version 4 (v4) is a validated candidate split format. However, because it relies on the legacy `preprocessed/` keypoint caches whose exact generation provenance is unresolved, v4 is not yet fully provenance-complete or publication-ready.

This pipeline implements a signer-independent evaluation protocol for ISL-CSLRT
using Leave-One-Signer-Out (LOSO) cross-validation. Whether prior published
work has applied a comparable protocol to this dataset has not been verified
by a systematic literature review.

The standard split (Signers 3-6 train / Signer 7 val / Signers 1-2 test)
is a particularly challenging generalisation setting: training on HD data,
testing on low-resolution mobile footage from unseen signers.
