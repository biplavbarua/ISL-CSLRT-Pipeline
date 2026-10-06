import os, cv2, numpy as np

out_dir = '../preprocessed_v4/plots'

# 1. Zero clip
frames_dir1 = '../ISL_CSLRT_Corpus/Frames_Sentence_Level/he is on the way/3'
files1 = sorted([f for f in os.listdir(frames_dir1) if f.endswith('.jpg') or f.endswith('.png')])
indices1 = np.linspace(0, len(files1)-1, 3, dtype=int)

for i, idx in enumerate(indices1):
    img = cv2.imread(os.path.join(frames_dir1, files1[idx]))
    if img is not None:
        cv2.imwrite(os.path.join(out_dir, f'zero_clip_{i}.png'), img)

# 2. Max gap clip (forward filled for 24 frames)
frames_dir2 = '../ISL_CSLRT_Corpus/Frames_Sentence_Level/how are things/5'
files2 = sorted([f for f in os.listdir(frames_dir2) if f.endswith('.jpg') or f.endswith('.png')])
coords_path2 = '../preprocessed_v4/keypoints_raw/how_are_things/5/coords.npy'
coords2 = np.load(coords_path2)

# Find first valid frame (frame 24, i.e., index 24)
first_valid_frame = 24
if first_valid_frame < len(files2):
    valid_coords = coords2[first_valid_frame, :75, :]
else:
    valid_coords = coords2[-1, :75, :]

indices2 = [0, 12, 23] # During the 24-frame gap

for i, idx in enumerate(indices2):
    img = cv2.imread(os.path.join(frames_dir2, files2[idx]))
    if img is not None:
        h, w = img.shape[:2]
        # Draw forward-filled coords
        for j in range(75):
            cx, cy = int(valid_coords[j, 0] * w), int(valid_coords[j, 1] * h)
            if cx == 0 and cy == 0: continue
            color = (0, 255, 0) if j < 33 else (0, 165, 255)
            cv2.circle(img, (cx, cy), 3, color, -1)
        cv2.imwrite(os.path.join(out_dir, f'gap_clip_{i}.png'), img)
        
print("Render complete.")
