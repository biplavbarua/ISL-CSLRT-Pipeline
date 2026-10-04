# Data Quality & Validation Report

This document outlines key data quality findings discovered during the pipeline validation phase, specifically concerning MediaPipe's detection robustness on this dataset.

## 1. Hand Detection: Mobile vs HD Footage
Contrary to the initial hypothesis that low-resolution mobile footage (Signers 1 & 2) would suffer from poor detection due to motion blur, the empirical data shows the exact opposite. 

**Per-Signer Hand Detection Rates (Mean % of frames with hands detected):**
* **Signer 2 (Mobile):** 93.4% (Best)
* **Signer 1 (Mobile):** 83.1% 
* **Signer 7 (HD Studio):** 77.2%
* **Signer 6 (HD Studio):** 70.9%
* **Signer 5 (HD Studio):** 60.5%
* **Signer 4 (HD Studio):** 60.0%
* **Signer 3 (HD Studio):** 54.9% (Worst)

**Conclusion:** The HD studio footage systematically fails to track hands far more often than the low-res mobile footage. While the pose landmarker tracked the signer's body in >99.9% of all frames regardless of camera, the hand landmarker struggled heavily on Signers 3-7. This is likely due to environmental factors in the HD studio (e.g., poor contrast between hands and clothing/background, or tight framing causing hands to exit the frame).

## 2. Confidence Metric Clarification
The `confidence.npy` output array behaves differently for Pose vs Hands:
* **Pose:** Outputs a continuous visibility probability [0.0, 1.0].
* **Hands:** The MediaPipe Tasks API does not provide per-landmark visibility for hands. The pipeline hardcodes hand confidence to exactly `1.0` if the hand is detected in the frame, and `0.0` if it is completely missed. It functions as a binary `presence` flag.

## 3. The 0% Detection Clip
* **Clip:** `he_is_on_the_way/3`
* **Finding:** Hand detection was exactly 0% for the entire duration. Visual inspection confirms the signer is sitting perfectly still with his hands resting in his lap, entirely off-camera. 
* **Action Required:** This clip contains no sign language content and should ideally be excluded from training in the `clip_index.csv`.

## 4. Interpolation Limitations on Long Gaps
The pipeline uses linear interpolation (and forward/backward filling at boundaries) to bridge gaps where MediaPipe loses tracking (Step 02).
* **Short Gaps (Median = 1 frame, Mean = 2.3 frames):** The vast majority of missing detections are 1-3 frame flickers. Interpolation bridges these perfectly.
* **Long Gaps (Max = 24 frames):** In clip `how_are_things/5`, there is a 24-frame gap at the start of the video. Visual inspection reveals the signer rapidly bringing her hands up, causing severe motion blur that MediaPipe failed to track. Because this gap occurred at the start of the clip, Step 02 **forward-filled** the coordinates from frame 24 backwards to frame 0. 
* **Limitation:** The forward-filled hand coordinates hover statically in mid-air for 24 frames, completely ignoring the signer's actual fast, curved gesture. Interpolation on long gaps (>5-10 frames) fabricates data that does not represent the ground truth, which models may erroneously learn from.
