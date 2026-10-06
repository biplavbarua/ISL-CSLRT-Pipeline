# ISL-CSLRT Training

This directory contains the neural network architecture and training scripts for the ISL-CSLRT dataset.

## Feature Engineering: The 126-Landmark Subset

Instead of passing all 543 MediaPipe Holistic landmarks into the model (which would be largely redundant and noisy, especially for the face), `dataset.py` explicitly extracts a tailored subset of **126 landmarks**:

| Group | Indices | Count | Rationale |
|---|---|---|---|
| **Pose** | 0 – 32 | 33 | Captures upper body posture, arms, shoulders, and head orientation. |
| **Left Hand** | 33 – 53 | 21 | Full articulation of the non-dominant hand. |
| **Right Hand** | 54 – 74 | 21 | Full articulation of the dominant hand. |
| **Face (Lips)** | [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415] (+75 offset) | 39 | Crucial for capturing mouth morphemes, lip rounding, and jaw drop in non-manual sign language grammar. |
| **Face (Eyebrows)** | [46, 52, 53, 55, 65, 70, 276, 282, 283, 285, 295, 300] (+75 offset) | 12 | Essential for grammatical markers (e.g., raised eyebrows for yes/no questions, furrowed for wh-questions). |
| **Total** | | **126** | |

This trims the input feature size from 1,629 coordinates (543 * 3) down to just **378 coordinates** (126 * 3) per frame, drastically reducing the parameter count of the first embedding layer and filtering out irrelevant facial mesh points (like the cheeks and nose ridge) that carry no linguistic signal.
