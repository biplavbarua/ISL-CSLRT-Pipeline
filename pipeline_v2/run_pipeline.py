"""
pipeline_v2/run_pipeline.py

Master runner — executes the full v2 pipeline end to end.

Usage:
    cd ISL-CSLRT-Pipeline
    python3 pipeline_v2/run_pipeline.py [--limit N] [--skip_existing]
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

def run_step(name, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"  ✓ {name} completed in {elapsed/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL-CSLRT v2 Full Pipeline")
    parser.add_argument("--limit",        type=int,  default=None,
                        help="Only process first N sentence classes (for testing)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip already-processed clips (default: True)")
    parser.add_argument("--start_at",     type=int,  default=1,
                        help="Start at step N (1=extract, 2=normalize, 3=segment, 4=splits, 5=export)")
    parser.add_argument("--stop_at",      type=int,  default=5,
                        help="Stop after step N")
    args = parser.parse_args()

    import step01_extract_from_video as s1
    import step02_normalize           as s2
    import step03_segment             as s3
    import step04_build_splits        as s4
    import step05_export              as s5

    steps = {
        1: ("Step 01 — Extract keypoints from videos",
            lambda: s1.process_all(limit=args.limit, skip_existing=args.skip_existing)),
        2: ("Step 02 — Normalize keypoints",
            lambda: s2.process_all(skip_existing=args.skip_existing)),
        3: ("Step 03 — Temporal sampling (T=32 segments)",
            lambda: s3.process_all(skip_existing=args.skip_existing)),
        4: ("Step 04 — Build dataset splits",
            s4.process_all),
        5: ("Step 05 — Export to HDF5",
            s5.process_all),
    }

    for step_num in range(args.start_at, args.stop_at + 1):
        name, fn = steps[step_num]
        run_step(name, fn)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Generate keyframe report:")
    print("       python3 pipeline_v2/report_keyframes.py")
    print("\n  2. Train all 3 models:")
    print("       python3 pipeline_v2/models/train_all.py \\")
    print("         --exports_dir preprocessed_v5/exports \\")
    print("         --device mps --epochs 100")
    print("\n  3. (Optional) Start Ollama for LLM word prediction:")
    print("       ollama serve          # in a separate terminal")
    print("       ollama pull gemma2:2b # first time only")
    print("       python3 pipeline_v2/llm_decoder.py")

