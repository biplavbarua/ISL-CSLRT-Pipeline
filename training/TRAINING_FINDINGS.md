# Training Findings: Signer Generalization vs Memorization

This document summarizes the modeling and diagnostic investigations conducted to
determine the viability of training a 97-way sign language recognition model given
the current dataset volume (662 total usable clips, ~3.9 examples per class in the
standard train split — one per training signer; each signer records each sentence
once).

## 1. Signer-Independent Configurations (Cross-Signer Transfer)

We ran 7 distinct architectural and hyperparameter configurations on the strict
signer-independent split (`standard_train`, `standard_val`, `standard_test`) to
evaluate true generalization to unseen mobile signers.

> **Note on git hash and code provenance.** All six softmax runs (Baseline through
> Combined) share the `results.json` field `git_hash: fe7e0b8`, but their code
> *differed at run time*. `training/config.py` did not exist until commit `3988874`;
> each run was driven by a separate root-level script that hard-coded its own
> hyperparameter overrides. The git hash therefore does **not** capture uncommitted
> edits that were active when steps 2–5 ran.

| Configuration | Best Val Acc | Test Acc | Best Epoch | What differs from baseline | Notes on Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** (BiGRU, H=256) | 4.30% | 3.13% | 18 | — | Best of the softmax runs on test; 100 epochs, no early stopping. |
| **Early Stopping** (Patience=15) | 5.38% | 2.08% | 24 | EarlyStopping patience=15 added | Stopped at epoch 24; higher val acc than baseline but lower test acc. |
| **Reduced Capacity** (H=64) | 2.15% | 0.52% | 3 | hidden_size 256→64 | Immediate loss floor, failed to learn. |
| **Augmentation** (Spatial Jitter) | 1.08% | 1.04% | 1 | Gaussian noise σ=0.01 added to coords | Hit loss floor by epoch 1. |
| **No-Face Landmarks** | 1.08% | 1.56% | 3 | Face landmarks dropped (input 378→198) | Loss floor; dropping face made no measurable difference. |
| **Combined** (Early Stop + H=64) | 2.15% | 2.08% | 9 | EarlyStopping + hidden_size=64 | Prevented metric collapse but flatlined at random-guess loss. |
| **Prototype Network** (Metric) | 7.53% | 2.08% | 26 | ProtoNet loss, episodic sampling | Separated classes on train set (25%+ acc by epoch 5, >90% at end) but severely overfit. |

All val/test accuracies are sourced directly from the corresponding `results.json`
log files. Random-chance baseline for 97 classes ≈ 1.03%.

### The Loss-Floor Phenomenon
For the softmax classification models, 3 of the configurations (Reduced Capacity,
Augmentation, No-Face) completely failed to leave the random-guess loss floor. For a
97-way classifier, the expected cross-entropy loss for random guessing is
$\ln(97) \approx 4.575$. These models' training and validation losses hovered
identically at $\sim 4.57$ across their first 5 epochs, meaning the network couldn't
find even a weak local gradient to pull classes apart.

## 2. Prototype Geometry Diagnostics (Overfitting Confirmation)

To test whether the Prototype Network learned *any* meaningful geometry that just fell
short of the Top-1 threshold, we measured the actual embedding distances (intra-class
vs inter-class) on both familiar and unseen signers:

- **Train Set (Leave-One-Out, Familiar Signers):**
  - Mean intra-class distance: **0.0179**
  - Mean inter-class distance: **2.0125**
  - *Result:* The metric space works phenomenally well on the training data. The
    network collapses familiar sequences to their true class prototypes with 100×
    tighter clustering than the distance to other classes.

- **Test Set (Unseen Signers):**
  - Mean intra-class distance: **1.8209**
  - Mean inter-class distance: **1.9597**
  - Mean ratio (nearest overall prototype / true prototype): **0.1071**
  - *Result:* Total geometry collapse on unseen signers. The intra-class distance is
    barely smaller than the inter-class distance (both are near 2.0, meaning the
    embeddings are effectively orthogonal). The ratio of 0.1071 reveals that a test
    clip is typically 10× closer to an *incorrect* prototype than its true one. The
    model memorized the specific physical traits of the training signers.

## 3. Pooled Random Split (NOT a same-signer test)

**Important clarification.** Because each signer performs each sentence exactly once,
any clip-level random split places some clips from every signer into every partition.
The 70/15/15 split tested here differs from the standard signer-independent split in
three ways:

1. **More train examples per class:** ~4.8 (463 clips / 97 classes) vs ~3.9 in the
   standard train split.
2. **Test signers' signing style is seen during training:** each signer appears in all
   three partitions, so the model can pattern-match on personal signing style.
3. **No HD-to-mobile distribution shift:** the standard split deliberately puts HD
   signers in train and mobile signers in test; the pooled split mixes them randomly.

These three differences make the pooled split easier than the standard split.

**Setup:** All 662 clips were pooled, shuffled with `np.random.seed(42)`, and split
70/15/15. We ran the plain baseline softmax classifier (BiGRU, H=256, 100 epochs, no
early stopping) on this split.

**Result (from `results.json`):**
```json
{
  "best_val_acc": 8.080808080808081,
  "test_acc": 9.0,
  "training_time_minutes": 6.775424698988597
}
```
`best_epoch` is logged as `-1` because `train_model()` does not return the epoch
number; the best checkpoint was reloaded before the test evaluation.

The test set contained **100 clips**; 9.0% corresponds to **9/100 clips correct**
(~8.7× random chance). This shows the task is weakly learnable with this feature
representation, but the ceiling is low even without the cross-signer constraint.

## 4. Floor Check: Non-Parametric Baselines (BLOCKED — data unavailable)

> **Status:** The `.h5` exports (`preprocessed_v4/exports/`) are not present on the
> current machine. Results will be filled in when the data is available.

**Planned method:** Mean-pool each clip over 32 frames → (N, 378) feature vector.
Z-score with training-set mean and std (numpy only). Evaluate on the standard split
(Train=377, Val=93, Test=192, 97 classes).

| Method | Val correct/total | Val % | Test correct/total | Test % |
| :--- | :--- | :--- | :--- | :--- |
| Nearest-centroid (Euclidean) | —/93 | — | —/192 | — |
| Nearest-centroid (Cosine) | —/93 | — | —/192 | — |
| 1-NN (Cosine) | —/93 | — | —/192 | — |

*Random chance: ~1/97 ≈ 1.03%.*

## 5. Signer-Scaling Experiment (BLOCKED — data unavailable)

> **Status:** LOSO fold `.h5` files are not present on the current machine. Results
> will be filled in when the data is available.

**Planned method:** LOSO folds 1, 4, 7 (test signers 1, 4, 7). For k ∈ {1,2,3,4,5}
training signers per fold, train the baseline BiGRU (hidden=256, dropout=0.5, AdamW
lr=1e-3, wd=1e-4, cosine annealing, batch=32, grad-clip=1.0, early-stopping
patience=15, max 60 epochs). For k<5: 2 random signer subsets per fold (fixed seeds).
Select by val accuracy; reload best checkpoint for test.

| k signers | Runs | Mean test acc | Min test acc | Max test acc | Pooled correct/total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 6 | — | — | — | —/— |
| 2 | 6 | — | — | — | —/— |
| 3 | 6 | — | — | — | —/— |
| 4 | 6 | — | — | — | —/— |
| 5 | 3 | — | — | — | —/— |

*Signer IDs per clip are read from the `signer_id` field stored in each LOSO fold
`.h5` file, written by `step04_build_splits.py` from the source directory name
`Signer_N/`.*

## Conclusion

Signer-independent generalization on this corpus is infeasible given the current
data volume (~3.9 examples per class in the standard train split — one per training
signer; each signer records each sentence once). The models either fail to converge
entirely (staying at random-guess loss) or successfully memorize the training signers'
exact geometry without generalizing to unseen test signers. The pooled-split check
(9/100 clips correct, 9.0%) shows the task is weakly learnable but with a hard
ceiling on the current feature representation. This strongly points to a fundamental
data-volume ceiling rather than a hyperparameter or architectural mismatch.
