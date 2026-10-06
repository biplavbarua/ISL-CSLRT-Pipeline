import glob, numpy as np, itertools

files = glob.glob('../preprocessed_v4/keypoints_raw/*/*/confidence.npy')
clips_with_long_gaps = []

for f in files:
    conf = np.load(f)
    if conf.shape[0] == 0: continue
    
    det = conf[:, 33:75].max(axis=1) > 0.5
    
    long_gap_found = False
    for is_det, run in itertools.groupby(det):
        if not is_det and sum(1 for _ in run) > 8:
            long_gap_found = True
            break
            
    if long_gap_found:
        clips_with_long_gaps.append(f)

print(f"Clips with gaps > 8 frames: {len(clips_with_long_gaps)}")
for c in clips_with_long_gaps:
    print(c)
