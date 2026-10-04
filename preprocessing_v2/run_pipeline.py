"""
run_pipeline.py — Execute the Full Preprocessing Pipeline
==========================================================

Run this single script to execute all 5 steps in order:
  Step 00: Audit & index the raw dataset
  Step 01: Extract MediaPipe keypoints from all frames
  Step 02: Normalize keypoints + interpolate missing detections
  Step 03: Uniform temporal sampling to T=32 frames
  Step 04: Build signer-independent data splits (LOSO + standard)
  Step 05: Export to HDF5 + run verification checks

Usage:
  python3 run_pipeline.py              # run all steps
  python3 run_pipeline.py --steps 0 1  # run only steps 0 and 1
  python3 run_pipeline.py --steps 2 3 4 5  # resume from step 2

This is useful when Step 01 (extraction) has already been run and you want
to re-run normalization with different parameters without re-extracting.
"""

import argparse
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def run_step(step_num: int):
    t0 = time.time()
    if step_num == 0:
        log.info("\n" + "█" * 50)
        log.info("█  STEP 00 — AUDIT & CLEAN")
        log.info("█" * 50)
        from step00_clean import run_audit
        run_audit()

    elif step_num == 1:
        log.info("\n" + "█" * 50)
        log.info("█  STEP 01 — KEYPOINT EXTRACTION  (slowest step ~5-30min)")
        log.info("█" * 50)
        from step01_extract_keypoints import run_extraction
        run_extraction()

    elif step_num == 2:
        log.info("\n" + "█" * 50)
        log.info("█  STEP 02 — NORMALIZATION + INTERPOLATION")
        log.info("█" * 50)
        from step02_normalize import run_normalization
        run_normalization()

    elif step_num == 3:
        log.info("\n" + "█" * 50)
        log.info("█  STEP 03 — TEMPORAL SEGMENTATION")
        log.info("█" * 50)
        from step03_segment import run_segmentation
        run_segmentation()

    elif step_num == 4:
        log.info("\n" + "█" * 50)
        log.info("█  STEP 04 — BUILD SIGNER-INDEPENDENT SPLITS")
        log.info("█" * 50)
        from step04_build_splits import run_splits
        run_splits()

    elif step_num == 5:
        log.info("\n" + "█" * 50)
        log.info("█  STEP 05 — EXPORT + VERIFICATION")
        log.info("█" * 50)
        from step05_export import run_export
        run_export()

    elapsed = time.time() - t0
    log.info(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="ISL-CSLRT Preprocessing Pipeline"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Which steps to run (default: all). E.g. --steps 0 1"
    )
    args = parser.parse_args()

    t_total = time.time()
    log.info("=" * 60)
    log.info("  ISL-CSLRT PREPROCESSING PIPELINE")
    log.info(f"  Running steps: {args.steps}")
    log.info("=" * 60)

    for step in args.steps:
        run_step(step)

    total_elapsed = time.time() - t_total
    log.info("\n" + "=" * 60)
    log.info(f"  PIPELINE COMPLETE in {total_elapsed/60:.1f} minutes")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
