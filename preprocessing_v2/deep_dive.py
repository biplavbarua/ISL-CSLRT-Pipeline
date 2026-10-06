import glob, numpy as np
from collections import defaultdict
import itertools

files = glob.glob('../preprocessed_v4/keypoints_raw/*/*/confidence.npy')

# 1. PER-SIGNER BREAKDOWN & 2. THRESHOLD SENSITIVITY
signer_stats = defaultdict(list)
thresholds = [0.3, 0.4, 0.5, 0.6]
under_80_by_thresh = {t: 0 for t in thresholds}

# 3. GAP-LENGTH DISTRIBUTION
all_gaps = []
zero_det_clips = 0
partial_det_clips = 0

for f in files:
    signer = f.split('/')[-2]
    conf = np.load(f)
    n_frames = conf.shape[0]
    if n_frames == 0: continue
    
    hand_conf = conf[:, 33:75] # (n_frames, 42)
    max_conf = hand_conf.max(axis=1) # (n_frames,)
    
    # 1. Per-signer (using 0.5 threshold)
    det_05 = max_conf > 0.5
    pct_05 = det_05.sum() / n_frames * 100
    signer_stats[signer].append(pct_05)
    
    # 2. Thresholds
    for t in thresholds:
        pct = (max_conf > t).sum() / n_frames * 100
        if pct < 80:
            under_80_by_thresh[t] += 1
            
    # 3. Gaps
    if det_05.sum() == 0:
        zero_det_clips += 1
    else:
        partial_det_clips += 1
        # Calculate gaps (runs of False in det_05)
        # Using itertools.groupby
        for is_det, run in itertools.groupby(det_05):
            if not is_det:
                length = sum(1 for _ in run)
                all_gaps.append(length)

print("--- 1. PER-SIGNER BREAKDOWN (Threshold 0.5) ---")
for s in sorted(signer_stats.keys(), key=int):
    pcts = signer_stats[s]
    under_80 = sum(1 for p in pcts if p < 80)
    print(f"Signer {s}: Mean {np.mean(pcts):.1f}% | <80%: {under_80}/{len(pcts)}")

print("\n--- 2. THRESHOLD SENSITIVITY ---")
for t in thresholds:
    print(f"Threshold {t:.1f}: {under_80_by_thresh[t]} clips under 80% det.")

print("\n--- 3. GAP-LENGTH DISTRIBUTION (Threshold 0.5) ---")
print(f"Clips with exactly 0% detection (nothing to interpolate): {zero_det_clips}")
print(f"Clips with partial detection (scattered gaps): {partial_det_clips}")
if all_gaps:
    print(f"Gaps - Mean: {np.mean(all_gaps):.1f} frames | Median: {np.median(all_gaps):.1f} | Max: {np.max(all_gaps)}")
