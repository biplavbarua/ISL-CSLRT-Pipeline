import glob, json, numpy as np, itertools
from collections import Counter, defaultdict

files = glob.glob('../preprocessed_v4/keypoints_raw/*/*/confidence.npy')
long_gap_clips = []

for f in files:
    conf = np.load(f)
    if conf.shape[0] == 0: continue
    det = conf[:, 33:75].max(axis=1) > 0.5
    for is_det, run in itertools.groupby(det):
        if not is_det and sum(1 for _ in run) > 8:
            parts = f.split('/')
            sentence = parts[-3]
            signer = parts[-2]
            long_gap_clips.append((sentence, signer))
            break

print(f"Total long gap clips: {len(long_gap_clips)}\n")

# 1. Count per signer
print("--- Count per Signer ---")
signer_counts = Counter([c[1] for c in long_gap_clips])
for s in sorted(signer_counts.keys(), key=int):
    print(f"Signer {s}: {signer_counts[s]} clips")

# 2. Sentence clustering
print("\n--- Sentence Clustering ---")
sentence_counts = Counter([c[0] for c in long_gap_clips])
clustered = {s: c for s, c in sentence_counts.items() if c >= 2}
for s, c in sorted(clustered.items(), key=lambda x: -x[1]):
    print(f"'{s}': {c} clips")

# 3. Fraction of splits
print("\n--- Fraction of Splits ---")
long_gap_set = {f"{s}/{signer}" for s, signer in long_gap_clips}

with open('../preprocessed_v4/logs/split_summary.json') as f:
    splits = json.load(f)

import csv
with open('../preprocessed_v4/logs/segment_manifest.csv') as f:
    manifest = list(csv.DictReader(f))

# Helper to calculate fractions
def calc_fraction(split_name, condition):
    # Get all clips in this split
    # Wait, the summary JSON just has stats, not the exact clips.
    # It's easier to recreate the logic or read the CSVs from splits/.
    pass

import os
splits_dir = '../preprocessed_v4/splits'
for split_file in sorted(os.listdir(splits_dir)):
    if not split_file.endswith('.csv'): continue
    
    with open(os.path.join(splits_dir, split_file)) as f:
        rows = list(csv.DictReader(f))
    
    total = len(rows)
    if total == 0: continue
    
    count_long = sum(1 for r in rows if f"{r['sentence_slug']}/{r['signer_id']}" in long_gap_set)
    print(f"{split_file}: {count_long}/{total} clips ({count_long/total*100:.1f}%)")

