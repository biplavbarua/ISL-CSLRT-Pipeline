"""
pipeline_v2/step01_extract_from_video.py

Step 01 — Frame extraction directly from .mp4/.MP4 videos + MediaPipe Tasks API
keypoint extraction (3 separate landmarkers: Pose, Hand, Face).

NOTE: MediaPipe 0.10.x on Python 3.13 no longer ships mp.solutions.holistic.
We use three independent Tasks API landmarkers, assembling output into the
same 543-landmark array schema used throughout the rest of the pipeline.

Model files are auto-downloaded on first run to preprocessed_v5/models/.

Output per clip:
    preprocessed_v5/keypoints_raw/<sentence_slug>/<signer_id>/
        coords.npy       — (n_frames, 543, 3)  float32
        confidence.npy   — (n_frames, 543)     float32
        meta.json        — fps, total_frames, video_path, detection_stats

Also writes:
    preprocessed_v5/logs/keyframe_report.csv   — one row per clip
"""

import os
import re
import sys
import json
import time
import csv
import argparse
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    RunningMode,
)

sys.path.insert(0, os.path.dirname(__file__))
import config

# ─── Model download URLs ───────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(config.OUTPUT_ROOT), "preprocessed_v5", "models")
MODELS = {
    "pose": {
        "url":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "path": os.path.join(MODELS_DIR, "pose_landmarker_full.task"),
    },
    "hand": {
        "url":  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "path": os.path.join(MODELS_DIR, "hand_landmarker.task"),
    },
    "face": {
        "url":  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "path": os.path.join(MODELS_DIR, "face_landmarker.task"),
    },
}
N_FACE_USE = 468  # FaceLandmarker returns 478; use first 468 to match 543 total


# ─── Utilities ─────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def get_video_files(sentence_dir: str) -> list:
    exts = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}
    return [
        os.path.join(sentence_dir, f)
        for f in sorted(os.listdir(sentence_dir))
        if os.path.splitext(f)[1] in exts
    ]


# ─── Model download ────────────────────────────────────────────────────────────

def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, info in MODELS.items():
        path = info["path"]
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            print(f"  [model] '{name}' already present ({os.path.getsize(path)//1024} KB)")
            continue
        print(f"  [model] Downloading '{name}' → {path}")
        urllib.request.urlretrieve(info["url"], path)
        print(f"  [model] Done ({os.path.getsize(path)/1e6:.1f} MB)")


# ─── Landmarker initialisation ─────────────────────────────────────────────────

def init_landmarkers():
    pose_det = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELS["pose"]["path"]),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    ))
    hand_det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELS["hand"]["path"]),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    face_det = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELS["face"]["path"]),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    ))
    return pose_det, hand_det, face_det


# ─── Per-frame landmark extraction ────────────────────────────────────────────

def extract_landmarks(pose_det, hand_det, face_det, img_bgr: np.ndarray):
    """
    Run all three landmarkers on one BGR frame.

    Returns
    -------
    coords     : (543, 3)  float32
    confidence : (543,)    float32
    hit_pose   : bool
    hit_lhand  : bool
    hit_rhand  : bool
    """
    N = config.N_LANDMARKS
    coords     = np.zeros((N, 3), dtype=np.float32)
    confidence = np.zeros(N,      dtype=np.float32)

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    hit_pose = hit_lhand = hit_rhand = False

    # ── Pose (indices 0–32) ───────────────────────────────────────────────────
    pose_result = pose_det.detect(mp_image)
    if pose_result.pose_landmarks:
        hit_pose = True
        for i, lm in enumerate(pose_result.pose_landmarks[0]):
            coords[i]     = [lm.x, lm.y, lm.z]
            confidence[i] = lm.visibility

    # ── Hands (indices 33–74) ─────────────────────────────────────────────────
    hand_result = hand_det.detect(mp_image)
    for hi, hand_lms in enumerate(hand_result.hand_landmarks):
        handedness = hand_result.handedness[hi][0].category_name  # "Left" or "Right"
        if handedness == "Left":
            base = config.N_POSE           # 33
            hit_lhand = True
        else:
            base = config.N_POSE + config.N_LHAND   # 54
            hit_rhand = True
        for i, lm in enumerate(hand_lms):
            coords[base + i]     = [lm.x, lm.y, lm.z]
            confidence[base + i] = 1.0    # Tasks API: binary presence for hands

    # ── Face (indices 75–542) ─────────────────────────────────────────────────
    face_result = face_det.detect(mp_image)
    if face_result.face_landmarks:
        base = config.N_POSE + config.N_LHAND + config.N_RHAND  # 75
        for i, lm in enumerate(face_result.face_landmarks[0][:N_FACE_USE]):
            coords[base + i]     = [lm.x, lm.y, lm.z]
            confidence[base + i] = getattr(lm, "visibility", 1.0)

    return coords, confidence, hit_pose, hit_lhand, hit_rhand


# ─── Per-video extraction ──────────────────────────────────────────────────────

def extract_video(video_path: str, pose_det, hand_det, face_det):
    """
    Decode all frames from a video and run landmark extraction on each.

    Returns
    -------
    coords     : (n_frames, 543, 3)
    confidence : (n_frames, 543)
    stats      : dict
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    all_c, all_conf = [], []
    pose_hits = lhand_hits = rhand_hits = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        c, conf, hp, hl, hr = extract_landmarks(pose_det, hand_det, face_det, frame)
        all_c.append(c)
        all_conf.append(conf)
        if hp: pose_hits  += 1
        if hl: lhand_hits += 1
        if hr: rhand_hits += 1

    cap.release()

    n = len(all_c)
    if n == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    coords     = np.stack(all_c,    axis=0)
    confidence = np.stack(all_conf, axis=0)
    stats = {
        "fps":             round(fps, 2),
        "total_frames":    n,
        "pose_detection":  round(pose_hits  / n * 100, 2),
        "lhand_detection": round(lhand_hits / n * 100, 2),
        "rhand_detection": round(rhand_hits / n * 100, 2),
    }
    return coords, confidence, stats


# ─── Main ──────────────────────────────────────────────────────────────────────

def process_all(start_idx=0, limit=None, skip_existing=True):
    os.makedirs(config.KP_RAW_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR,   exist_ok=True)

    print("Checking/downloading MediaPipe task model files...")
    ensure_models()

    print("\nInitialising landmarkers (Pose + Hand + Face)...")
    pose_det, hand_det, face_det = init_landmarkers()
    print("  Landmarkers ready.\n")

    report_path = os.path.join(config.LOGS_DIR, "keyframe_report.csv")
    fieldnames = ["sentence","signer_id","slug","video_file","total_frames","fps",
                  "pose_det_%","lhand_det_%","rhand_det_%","status"]
    
    # Load existing report rows to avoid duplicates or data loss
    report_dict = {}
    if os.path.exists(report_path):
        with open(report_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                report_dict[(r["slug"], str(r["signer_id"]))] = r

    all_sentence_dirs = sorted([
        d for d in os.listdir(config.VIDEOS_DIR)
        if os.path.isdir(os.path.join(config.VIDEOS_DIR, d))
    ])
    total_in_corpus = len(all_sentence_dirs)

    end_idx = total_in_corpus if limit is None else min(start_idx + limit, total_in_corpus)
    selected_dirs = all_sentence_dirs[start_idx:end_idx]

    print(f"Batch Range: Sentences [{start_idx} to {end_idx - 1}] of {total_in_corpus} ({len(selected_dirs)} sentences)")

    def flush_report():
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in report_dict.values():
                writer.writerow(row)

    total_clips = failed = 0

    for idx_offset, sent_name in enumerate(selected_dirs):
        global_idx = start_idx + idx_offset
        sent_dir  = os.path.join(config.VIDEOS_DIR, sent_name)
        sent_slug = slugify(sent_name)
        videos    = get_video_files(sent_dir)

        if not videos:
            print(f"  [SKIP] No videos: {sent_name}")
            continue

        print(f"\n[{global_idx + 1}/{total_in_corpus}] [{sent_slug}]  {len(videos)} video(s)")

        for signer_id, video_path in enumerate(videos, start=1):
            out_dir     = os.path.join(config.KP_RAW_DIR, sent_slug, str(signer_id))
            coords_path = os.path.join(out_dir, "coords.npy")
            key = (sent_slug, str(signer_id))

            if skip_existing and os.path.exists(coords_path):
                meta_path = os.path.join(out_dir, "meta.json")
                stats = {}
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    stats = meta.get("detection_stats", {})
                report_dict[key] = {
                    "sentence": sent_name, "signer_id": signer_id,
                    "slug": sent_slug, "video_file": os.path.basename(video_path),
                    "total_frames": stats.get("total_frames", -1),
                    "fps": stats.get("fps", -1),
                    "pose_det_%": stats.get("pose_detection", -1),
                    "lhand_det_%": stats.get("lhand_detection", -1),
                    "rhand_det_%": stats.get("rhand_detection", -1),
                    "status": "cached",
                }
                total_clips += 1
                print(f"  Signer {signer_id}: [CACHED]")
                continue

            t0 = time.time()
            try:
                coords, confidence, stats = extract_video(
                    video_path, pose_det, hand_det, face_det
                )
            except Exception as e:
                print(f"  Signer {signer_id}: [ERROR] {e}")
                failed += 1
                report_dict[key] = {
                    "sentence": sent_name, "signer_id": signer_id,
                    "slug": sent_slug, "video_file": os.path.basename(video_path),
                    "total_frames": -1, "fps": -1,
                    "pose_det_%": -1, "lhand_det_%": -1, "rhand_det_%": -1,
                    "status": f"failed: {e}",
                }
                flush_report()
                continue

            os.makedirs(out_dir, exist_ok=True)
            np.save(coords_path,                               coords)
            np.save(os.path.join(out_dir, "confidence.npy"),  confidence)
            meta = {
                "sentence": sent_name, "signer_id": signer_id,
                "slug": sent_slug, "video_path": video_path,
                "detection_stats": stats,
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            elapsed = time.time() - t0
            print(f"  Signer {signer_id}: frames={stats['total_frames']}  "
                  f"pose={stats['pose_detection']}%  "
                  f"lh={stats['lhand_detection']}%  "
                  f"rh={stats['rhand_detection']}%  ({elapsed:.1f}s)")

            report_dict[key] = {
                "sentence": sent_name, "signer_id": signer_id,
                "slug": sent_slug, "video_file": os.path.basename(video_path),
                "total_frames": stats["total_frames"], "fps": stats["fps"],
                "pose_det_%": stats["pose_detection"],
                "lhand_det_%": stats["lhand_detection"],
                "rhand_det_%": stats["rhand_detection"],
                "status": "ok",
            }
            total_clips += 1
            flush_report()

    flush_report()
    print(f"\n{'='*60}")
    print(f"Batch complete.  Clips={total_clips}  Failed={failed}")
    print(f"Total entries in report: {len(report_dict)}")
    print(f"Keyframe report → {report_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 01 — Extract keypoints from videos")
    parser.add_argument("--start_idx", type=int, default=0, help="Sentence index to start at (0-based)")
    parser.add_argument("--limit",     type=int, default=None, help="Number of sentence classes in this batch")
    parser.add_argument("--no_skip",   action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()
    process_all(start_idx=args.start_idx, limit=args.limit, skip_existing=not args.no_skip)
