import os, glob, numpy as np
import cv2

print('--- CONFIDENCE-BASED DETECTION STATS ---')
files = glob.glob('../preprocessed_v4/keypoints_raw/*/*/confidence.npy')
print(f'Found {len(files)} confidence files.')

pose_pcts = []
hand_pcts = []

for f in files:
    conf = np.load(f)
    n_frames = conf.shape[0]
    if n_frames == 0: continue
    
    # Pose: 0-32 (33 landmarks)
    pose_conf = conf[:, 0:33]
    pose_det = (pose_conf.mean(axis=1) > 0.5).sum() / n_frames * 100
    
    # Hand: 33-74 (42 landmarks)
    hand_conf = conf[:, 33:75]
    # For hand, either left or right being detected might be enough? 
    # Or both? We can check if ANY hand is detected (mean of max of left/right, or just max conf of any hand point > 0.5)
    # Let's say if max conf in hand > 0.5 it means at least one hand is detected.
    hand_det = (hand_conf.max(axis=1) > 0.5).sum() / n_frames * 100
    
    pose_pcts.append(pose_det)
    hand_pcts.append(hand_det)

if pose_pcts:
    print(f'Pose Det Mean: {np.mean(pose_pcts):.1f}% | Min: {np.min(pose_pcts):.1f}% | <80%: {sum(1 for p in pose_pcts if p < 80)}')
    print(f'Hand Det Mean: {np.mean(hand_pcts):.1f}% | Min: {np.min(hand_pcts):.1f}% | <80%: {sum(1 for p in hand_pcts if p < 80)}')

print('\n--- VISUAL SPOT CHECK ---')
clip_name = 'He_is_going_into_the_room'
signer = '1'
frames_dir = f'../ISL_CSLRT_Corpus/Frames_Sentence_Level/He is going into the room/{signer}'
coords_path = f'../preprocessed_v4/keypoints_raw/{clip_name}/{signer}/coords.npy'

frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
coords = np.load(coords_path) # (n_frames, 543, 3)

out_dir = '../preprocessed_v4/plots'
os.makedirs(out_dir, exist_ok=True)

# Pick 4 evenly spaced frames
indices = np.linspace(0, len(frame_files)-1, 4, dtype=int)
for i, idx in enumerate(indices):
    img_path = os.path.join(frames_dir, frame_files[idx])
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    
    frame_coords = coords[idx, :75, :] # pose and hands
    for j in range(75):
        cx, cy = int(frame_coords[j, 0] * w), int(frame_coords[j, 1] * h)
        if cx == 0 and cy == 0: continue # zero-filled missed detection
        color = (0, 255, 0) if j < 33 else (0, 165, 255) # green for pose, orange for hands
        cv2.circle(img, (cx, cy), 3, color, -1)
        
    out_path = os.path.join(out_dir, f'spot_check_{i}.png')
    cv2.imwrite(out_path, img)
    print(f'Saved {out_path}')
