import sys, glob, os, cv2, numpy as np
sys.path.append('../preprocessing_v2')
from step01_extract_keypoints import ensure_models, init_landmarkers, extract_landmarks

ensure_models()
pose_det, hand_det, face_det, mp = init_landmarkers()

clip_dir = "../ISL_CSLRT_Corpus/Frames_Sentence_Level/He_is_going_into_the_room/1"
all_frames = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
for i, f in enumerate(all_frames):
    img = cv2.imread(f)
    c, cf = extract_landmarks(pose_det, hand_det, face_det, mp, img)
    if c[54, 0] > 0:
        print(f"Frame {i} right wrist x: {c[54, 0]}")
