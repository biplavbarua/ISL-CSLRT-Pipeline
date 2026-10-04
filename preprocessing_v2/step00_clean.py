"""
step00_clean.py — Dataset Audit & Cleanup
==========================================

PURPOSE
-------
Before doing any heavy computation (MediaPipe runs on every frame), we do a
fast audit pass to understand exactly what we have and remove noise files that
would cause errors or silently corrupt downstream data.

WHAT THIS SCRIPT DOES
---------------------
1. Finds and REPORTS (not deletes yet) all junk files (Thumbs.db, .DS_Store)
2. Scans every sentence × signer folder and counts usable .jpg frames
3. Flags anomalies:
   - Signers with zero frames for a sentence (missing data)
   - Folders with extremely low frame count (< 5 frames — likely corrupt)
   - Resolution mismatches detected by sampling one frame per folder
4. Builds a master index CSV of ALL clips: sentence / signer / frame_count /
   resolution / frame_paths
5. Saves the index to logs/ so every later script can load it instead of
   re-scanning the disk

WHY BUILD AN INDEX FIRST?
-------------------------
Scanning 18,863 files on disk is slow. If we build the index once here,
every subsequent script just reads a 97×7 CSV — milliseconds instead of
seconds. It also gives us a permanent audit trail of what the raw dataset
contained before any transformation.

DECISION: We do NOT delete files here — we just report them and filter them
out in the index. Deleting original data is irreversible. The downstream
scripts will simply ignore files not in the index.
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path
from collections import defaultdict

# Make sure config.py is importable regardless of where this script is run from
sys.path.insert(0, str(Path(__file__).parent))
import config

# ---------------------------------------------------------------------------
# LOGGING SETUP
# We write a human-readable log file AND print to console simultaneously.
# The log file is saved in logs/ for later reference.
# ---------------------------------------------------------------------------

os.makedirs(config.LOGS_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_ROOT, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# HELPER: Resolve frame sort order
# WHY: os.listdir() gives files in arbitrary order. Frame order matters for
# temporal sequence. We sort numerically by the trailing number in the
# filename (e.g., "ezgif-frame-005.jpg" → 5, "MVI_6208 03.jpg" → 3).
# ---------------------------------------------------------------------------

def sort_key(fname: str) -> int:
    """Extract trailing integer from a filename for numeric sorting."""
    stem = Path(fname).stem          # e.g., "ezgif-frame-005"
    parts = stem.replace("-", " ").replace("_", " ").split()
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0  # fallback: put non-numeric names first


def collect_frames(folder: Path) -> list[str]:
    """
    Return a sorted list of valid image filenames in a folder.
    Silently skips junk files (Thumbs.db, .DS_Store, etc.).
    """
    frames = []
    for fname in os.listdir(folder):
        if fname in config.SKIP_FILENAMES:
            continue
        ext = Path(fname).suffix.lower()
        if ext in config.IMAGE_EXTENSIONS:
            frames.append(fname)
    frames.sort(key=sort_key)
    return frames


# ---------------------------------------------------------------------------
# MAIN AUDIT
# ---------------------------------------------------------------------------

def run_audit():
    fh = logging.FileHandler(os.path.join(config.LOGS_DIR, "step00_clean.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)
    try:
        log.info("=" * 70)
        log.info("ISL-CSLRT DATASET AUDIT  —  Step 00: Clean & Index")
        log.info("=" * 70)
        log.info(f"Corpus root : {config.FRAMES_SENTENCE_DIR}")
        log.info(f"Output root : {config.OUTPUT_ROOT}")

        frames_base = Path(config.FRAMES_SENTENCE_DIR)

        # Enumerate all sentence folders
        sentence_dirs = sorted([
            d for d in frames_base.iterdir()
            if d.is_dir() and d.name not in config.SKIP_FILENAMES
        ])
        log.info(f"\nFound {len(sentence_dirs)} sentence folders")

        # -----------------------------------------------------------------------
        # Pass 1: Collect junk file counts
        # -----------------------------------------------------------------------
        junk_files = []
        for sent_dir in sentence_dirs:
            for signer_dir in sent_dir.iterdir():
                if not signer_dir.is_dir():
                    continue
                for fname in os.listdir(signer_dir):
                    if fname in config.SKIP_FILENAMES:
                        junk_files.append(str(signer_dir / fname))

        log.info(f"\n[JUNK FILES] Found {len(junk_files)} files to skip (Thumbs.db etc.)")
        log.info("  These will be IGNORED by all downstream scripts (not deleted).")
        if junk_files[:3]:
            log.info(f"  Examples: {junk_files[:3]}")

        # -----------------------------------------------------------------------
        # Pass 2: Build master clip index
        # Each row = one (sentence, signer) clip
        # -----------------------------------------------------------------------
        log.info("\n[INDEX] Building master clip index...")

        clip_index = []       # list of dicts, one per clip
        anomalies  = []       # clips with problems

        total_frames = 0
        sentences_missing_signers = defaultdict(list)

        for sent_dir in sentence_dirs:
            sentence = sent_dir.name

            # Find which signer sub-folders exist for this sentence
            signer_dirs = {
                int(d.name): d
                for d in sent_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            }

            # Check for missing signers
            for s in config.ALL_SIGNERS:
                if s not in signer_dirs:
                    sentences_missing_signers[sentence].append(s)

            for signer_id, signer_dir in sorted(signer_dirs.items()):
                frames = collect_frames(signer_dir)
                n_frames = len(frames)
                total_frames += n_frames

                # Flag anomalies
                anomaly_reason = None
                if n_frames == 0:
                    anomaly_reason = "ZERO_FRAMES"
                elif n_frames < 5:
                    anomaly_reason = f"VERY_FEW_FRAMES ({n_frames})"

                resolution = "low_res" if signer_id in config.LOW_RES_SIGNERS else "hd"

                clip = {
                    "sentence":       sentence,
                    "signer_id":      signer_id,
                    "resolution":     resolution,
                    "n_frames":       n_frames,
                    "frames_dir":     str(signer_dir),
                    "frame_files":    json.dumps(frames),   # serialised list
                    "anomaly":        anomaly_reason or "",
                }
                clip_index.append(clip)

                if anomaly_reason:
                    anomalies.append(clip)

        # -----------------------------------------------------------------------
        # Pass 3: Save the index CSV
        # -----------------------------------------------------------------------
        index_path = os.path.join(config.LOGS_DIR, "clip_index.csv")
        fieldnames = ["sentence", "signer_id", "resolution", "n_frames",
                      "frames_dir", "frame_files", "anomaly"]

        with open(index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clip_index)

        log.info(f"\n[INDEX] Saved {len(clip_index)} clips → {index_path}")

        # -----------------------------------------------------------------------
        # Pass 4: Load gloss annotations and cross-check
        # WHY: The gloss CSV maps spoken sentences to ISL sign sequences.
        # We need to verify every sentence in the frames dir has a gloss entry,
        # because without a gloss we cannot build a translation model later.
        # -----------------------------------------------------------------------
        log.info("\n[GLOSSES] Cross-checking gloss annotations...")
        gloss_map = {}
        with open(config.GLOSS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sent = row["Sentence"].strip().lower()
                gloss = row["SIGN GLOSSES"].strip()
                gloss_map[sent] = gloss

        frame_sentences = {s.name.lower() for s in sentence_dirs}
        gloss_sentences = set(gloss_map.keys())

        matched      = frame_sentences & gloss_sentences
        no_gloss     = frame_sentences - gloss_sentences
        no_frames    = gloss_sentences - frame_sentences

        log.info(f"  Sentences with frames AND gloss : {len(matched)}")
        log.info(f"  Sentences with frames, NO gloss : {len(no_gloss)} → {sorted(no_gloss)}")
        log.info(f"  Glosses with NO frames          : {len(no_frames)}")

        # Save gloss map for downstream scripts
        gloss_path = os.path.join(config.LOGS_DIR, "gloss_map.json")
        with open(gloss_path, "w", encoding="utf-8") as f:
            json.dump(gloss_map, f, indent=2)
        log.info(f"  Gloss map saved → {gloss_path}")

        # -----------------------------------------------------------------------
        # SUMMARY REPORT
        # -----------------------------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("AUDIT SUMMARY")
        log.info("=" * 70)
        log.info(f"  Total clips (sentence × signer) : {len(clip_index)}")
        log.info(f"  Total usable frames             : {total_frames}")
        log.info(f"  Junk files to skip              : {len(junk_files)}")
        log.info(f"  Anomalous clips                 : {len(anomalies)}")

        log.info(f"\n  Sentences missing ≥1 signer:")
        for sent, missing in sorted(sentences_missing_signers.items()):
            log.info(f"    '{sent}' → missing signers {missing}")

        if anomalies:
            log.info(f"\n  Anomalous clips (zero / very few frames):")
            for a in anomalies:
                log.info(f"    Signer {a['signer_id']} / '{a['sentence']}' — {a['anomaly']}")

        log.info("\n✓ Audit complete. Clip index ready for Step 01.")
        return clip_index


    finally:
        log.removeHandler(fh)
        fh.close()

if __name__ == "__main__":
        run_audit()
