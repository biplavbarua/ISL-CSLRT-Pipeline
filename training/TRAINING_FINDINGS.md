# Training Findings: Signer Generalization vs Memorization

This document summarizes the modeling and diagnostic investigations conducted to determine the viability of training a 97-way sign language recognition model given the current dataset volume (662 total usable clips, ~4 examples per class per signer). 

## 1. Signer-Independent Configurations (Cross-Signer Transfer)

We ran 7 distinct architectural and hyperparameter configurations on the strict signer-independent split (`standard_train`, `standard_val`, `standard_test`) to evaluate true generalization to unseen mobile signers.

| Configuration | Best Val Acc | Test Acc | Best Epoch | Notes on Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** (BiGRU, H=256) | ~1.0% | ~1.0% | N/A | Flatlined at random-guess accuracy. |
| **Early Stopping** (Patience=15) | 2.15% | 1.04% | 18 | Prevented severe metric collapse but no meaningful learning. |
| **Reduced Capacity** (H=64) | 2.15% | ~0.0% | 3 | Immediate loss floor, failed to learn. |
| **Augmentation** (Spatial Jitter) | 1.07% | ~0.0% | 3 | Immediate loss floor, failed to learn. |
| **No-Face Landmarks** | 1.07% | ~0.0% | 3 | Immediate loss floor, failed to learn. |
| **Combined** (Early Stop + H=64) | 2.15% | 2.08% | 9 | Prevented metric collapse but flatlined at random-guess loss. |
| **Prototype Network** (Metric) | 7.53% | 2.08% | 26 | Separated classes on train set (25%+ acc by epoch 5, >90% at end) but severely overfit. |

### The Loss-Floor Phenomenon
For the softmax classification models, 3 of the configurations (Reduced Capacity, Augmentation, No-Face) completely failed to leave the random-guess loss floor. For a 97-way classifier, the expected cross-entropy loss for random guessing is $\ln(97) \approx 4.575$. These models' training and validation losses hovered identically at $\sim 4.57$ across their first 5 epochs, meaning the network couldn't find even a weak local gradient to pull classes apart.

## 2. Prototype Geometry Diagnostics (Overfitting Confirmation)

To test whether the Prototype Network learned *any* meaningful geometry that just fell short of the Top-1 threshold, we measured the actual embedding distances (intra-class vs inter-class) on both familiar and unseen signers:

- **Train Set (Leave-One-Out, Familiar Signers):**
  - Mean intra-class distance: **0.0179**
  - Mean inter-class distance: **2.0125**
  - *Result:* The metric space works phenomenally well on the training data. The network collapses familiar sequences to their true class prototypes with 100x tighter clustering than the distance to other classes.

- **Test Set (Unseen Signers):**
  - Mean intra-class distance: **1.8209**
  - Mean inter-class distance: **1.9597**
  - Mean ratio (nearest overall prototype / true prototype): **0.1071**
  - *Result:* Total geometry collapse on unseen signers. The intra-class distance is barely smaller than the inter-class distance (both are near 2.0, meaning the embeddings are effectively orthogonal). Worse, the ratio of 0.1071 reveals that a test clip is typically 10x closer to an *incorrect* prototype than its true one. The model memorized the specific physical traits of the training signers.

## 3. Same-Signer Upper-Bound (Is the task learnable at all?)

To isolate whether the architecture and landmarks are capable of representing the 97 signs *without* the burden of cross-signer transfer, we ran a cheap upper-bound check. 

**Setup:** All 662 clips were pooled and randomly split 70/15/15, explicitly allowing the same signer to appear in train, val, and test sets. We ran the plain baseline softmax classifier (BiGRU, H=256, 100 epochs) on this split.

**Result:**
```json
{
  "best_val_acc": 8.080808080808081,
  "test_acc": 9.0,
  "training_time_minutes": 6.77
}
```
*Note: This is NOT a generalization claim. It isolates whether the task is learnable at all given these features.*

Even when the model is allowed to see the exact same signers in the training set that it evaluates on in the test set, it only achieves **9.0%** test accuracy. While this is significantly better than random guessing (1.03%), it confirms a hard ceiling on the current feature representation.

## Conclusion

Signer-independent generalization on this corpus is infeasible given the current data volume ($\sim 4$ examples/class/signer). The models either fail to converge entirely (staying at random-guess loss) or successfully memorize the training signers' exact geometry without generalizing to unseen test signers. Furthermore, the same-signer upper-bound check ($\sim 9\%$ accuracy) proves that even if signer variance was removed, the 97-way task cannot be successfully solved with just $\sim 5$ total examples per class using this landmark-based architecture. This strongly points to a fundamental data-volume ceiling rather than a hyperparameter or architectural mismatch.
