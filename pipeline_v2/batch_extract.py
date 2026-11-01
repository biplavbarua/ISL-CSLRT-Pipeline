"""
pipeline_v2/batch_extract.py

Runs Step 01 video extraction in well-defined batches, reporting status after each batch.

Total sentences: 101
Batch 1: Sentences  0 – 19  (20 classes, ~135 clips)
Batch 2: Sentences 20 – 39  (20 classes, ~135 clips)
Batch 3: Sentences 40 – 59  (20 classes, ~135 clips)
Batch 4: Sentences 60 – 79  (20 classes, ~135 clips)
Batch 5: Sentences 80 – 100 (21 classes, ~140 clips)

Usage:
    # Run a specific batch:
    python3 pipeline_v2/batch_extract.py --batch 1

    # Run all batches sequentially:
    python3 pipeline_v2/batch_extract.py --batch all
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))
import config
from step01_extract_from_video import process_all

BATCH_SIZE = 20

def get_batches():
    all_sents = sorted([
        d for d in os.listdir(config.VIDEOS_DIR)
        if os.path.isdir(os.path.join(config.VIDEOS_DIR, d))
    ])
    total = len(all_sents)
    batches = []
    start = 0
    batch_num = 1
    while start < total:
        count = min(BATCH_SIZE, total - start)
        batches.append({
            "batch_num": batch_num,
            "start_idx": start,
            "limit":     count,
            "sentences": all_sents[start : start + count]
        })
        start += count
        batch_num += 1
    return batches, total


def run_single_batch(b):
    print("\n" + "="*70)
    print(f"STARTING BATCH {b['batch_num']}: Sentences {b['start_idx']} to {b['start_idx'] + b['limit'] - 1}")
    print(f"Classes: {b['sentences'][0]} ... to ... {b['sentences'][-1]}")
    print("="*70 + "\n")

    t0 = time.time()
    process_all(start_idx=b["start_idx"], limit=b["limit"], skip_existing=True)
    elapsed = time.time() - t0
    print(f"\n>>> Batch {b['batch_num']} finished in {elapsed/60:.2f} minutes.")


def main():
    parser = argparse.ArgumentParser(description="Run video extraction in batches")
    parser.add_argument("--batch", type=str, default="1",
                        help="Batch number (1, 2, 3, 4, 5) or 'all' (default: 1)")
    args = parser.parse_args()

    batches, total = get_batches()
    print(f"Total sentence classes: {total} across {len(batches)} batches.")

    if args.batch.lower() == "all":
        for b in batches:
            run_single_batch(b)
        print("\nAll batches completed successfully!")
    else:
        try:
            b_num = int(args.batch)
            if b_num < 1 or b_num > len(batches):
                print(f"Invalid batch {b_num}. Choose 1 to {len(batches)}.")
                return
            b = batches[b_num - 1]
            run_single_batch(b)
        except ValueError:
            print("Please specify a batch number (1-5) or 'all'.")


if __name__ == "__main__":
    main()

