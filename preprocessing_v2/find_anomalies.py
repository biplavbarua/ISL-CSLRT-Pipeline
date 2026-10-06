import glob, numpy as np, itertools

files = glob.glob('../preprocessed_v4/keypoints_raw/*/*/confidence.npy')
zero_clip = None
max_gap_clip = None
max_gap_len = 0
max_gap_start = 0

for f in files:
    conf = np.load(f)
    n_frames = conf.shape[0]
    if n_frames == 0: continue
    
    hand_conf = conf[:, 33:75]
    det = hand_conf.max(axis=1) > 0.5
    
    if det.sum() == 0:
        zero_clip = f
        
    runs = []
    idx = 0
    for is_det, run in itertools.groupby(det):
        length = sum(1 for _ in run)
        if not is_det and length > max_gap_len:
            max_gap_len = length
            max_gap_clip = f
            max_gap_start = idx
        idx += length

print(f"Zero clip: {zero_clip}")
print(f"Max gap clip: {max_gap_clip} | Length: {max_gap_len} | Starts at frame: {max_gap_start}")
