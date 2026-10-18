# Phase 2A Results Analysis

## Run Configuration
- Method: Gradient Descent
- MAX_EPOCHS: 500, PATIENCE: 100, LR: 0.01
- STEEPNESS: 50
- PARAM_BOUNDS: threshold_img (0.05, 0.50), threshold_seg (0.05, 0.50), alpha (0.0, 1.0)
- INITIAL_PARAMS: threshold_img=0.10, threshold_seg=0.25, alpha=0.5

---

## Training Curves

![Training Curves](results/training_curves.png)

### Final Parameters
| Parameter | Initial | Final |
|---|---|---|
| θ_img | 0.100 | 0.050 (hit minimum bound) |
| θ_seg | 0.250 | 0.050 (hit minimum bound) |
| α | 0.500 | 0.000 |

### Final Metrics
| Metric | Value | Clinical Meaning |
|---|---|---|
| Dice | 0.0823 ± 0.0395 | Only 8.2% overlap between predicted and annotated abnormal regions — very poor spatial agreement |
| IoU | 0.0433 ± 0.0222 | Intersection over Union of 4.3% — the predicted and annotated regions barely overlap |
| Precision | 0.0608 ± 0.0339 | Only 6% of pixels flagged as abnormal are actually annotated as abnormal — 94% are false positives |
| Recall | 0.1326 ± 0.0500 | The model captures 13.3% of the annotated abnormal pixels — misses 87% of what was annotated |

---

### Training Progress (top left)

- Train loss starts at ~0.95 and drops steadily to ~0.89 — the model is learning
- Val loss starts at ~0.96, drops to ~0.92 by epoch 30, then flattens and barely moves for the remaining 250 epochs
- Both losses are very high — Dice loss of 0.89 means Dice score of only 0.11
- There is a clear and persistent gap between train and val loss — train keeps slowly decreasing while val plateaus around 0.916. This is a sign of overfitting
- With only 3 scalar parameters and 16 training slices, overfitting shouldn't normally happen. But it is — because the training and validation cases have different annotation styles/sizes
- Training cases have very variable annotation sizes (1941–12789 pixels per slice) while validation cases are more consistent and smaller (1881–3778 pixels). The large training annotations (TL04051640 has 12,541 and 12,789 pixels) are pulling the thresholds down to flag more pixels
- The training is technically working but the val loss plateau tells you the current error maps simply don't contain enough localised signal to match the annotations precisely. The loss will not improve significantly without better input maps

---

### Image Error Threshold Evolution (top right)

- Starts at 0.090 (initial was 0.10, already moved by epoch 1)
- Drops very steeply between epoch 1 and ~epoch 10
- Hits the minimum bound of 0.05 by around epoch 10
- Stays completely flat at 0.05 for the remaining ~270 epochs
- θ_img collapsed even faster than θ_seg — it hit the minimum bound by epoch 10 while θ_seg took until epoch 25. This tells you the VAE image error map has even less useful localised signal than the seg uncertainty map
- The VAE image error map is diffuse — low-level error spread across the entire tissue region with no strong localised peaks at abnormal regions. At threshold 0.05, a huge portion of the tissue is flagged because almost every tissue pixel has error > 0.05
- The range goes from 0 to 0.425 but the distribution is heavily concentrated at low values — most pixels are between 0.00 and 0.10. So threshold 0.05 flags roughly half the tissue
- θ_img collapsed fastest because the VAE error map is the least localised of the two signals — it provides almost no spatial specificity for adenomyosis features

---

### Segmentation Uncertainty Threshold Evolution (bottom left)

- Starts at 0.235 (initial value was 0.25, already moved slightly by epoch 1)
- Drops very steeply between epoch 1 and ~epoch 20
- Hits the minimum bound of 0.05 by around epoch 25
- Stays flat at 0.05 for the remaining ~255 epochs
- The optimizer drove θ_seg to the minimum bound as fast as it could — at threshold 0.25, very few pixels pass → low recall → high Dice loss. Every time the threshold was lowered, more pixels were flagged → more true positives captured → Dice loss improved
- The seg uncertainty map has high uncertainty values along thin tissue boundary lines (the junctional zone). These lines are narrow — maybe 3-5 pixels wide. But even at threshold 0.05, the number of pixels above that threshold is much larger than the annotation size, so the prediction still over-covers the annotation
- The U-Net is uncertain along all tissue boundaries — not just the abnormal ones. It cannot distinguish between a normal junctional zone boundary and an abnormal one. The uncertainty signal is not specific enough to adenomyosis features — it fires everywhere there is a tissue boundary regardless of whether it is pathological
- θ_seg collapsed because the seg uncertainty map is not specific enough — it highlights all boundaries, not just abnormal ones

---

### Weighting Parameter α Evolution (bottom right)

- Starts at 0.5 (equal weight between both maps)
- Rises slightly to ~0.58 between epoch 1–30 — briefly favouring the VAE image error map
- Then drops steadily and continuously from epoch 30 to ~epoch 180
- Hits 0.0 at epoch 180 and stays flat for the remaining ~100 epochs

**Three phases:**

**Phase 1 (epoch 1–30): α rises to 0.58**
Both thresholds are still dropping. The optimizer briefly tries giving more weight to the VAE image error map — perhaps because at higher thresholds the VAE map was slightly more selective than the seg map.

**Phase 2 (epoch 30–180): α drops from 0.58 → 0.0**
Both thresholds have now hit their minimum bounds (0.05). The only remaining free parameter is α. The optimizer systematically reduces α — shifting weight away from the VAE map and towards the seg uncertainty map. Every small reduction in α slightly improves Dice loss, so it keeps going.

**Phase 3 (epoch 180–280): α = 0.0, completely flat**
The optimizer has fully abandoned the VAE image error map. The combined map is now 100% the seg uncertainty map. No further improvement is possible so early stopping triggers at epoch 280.

- The optimizer's journey tells a clear story: it tried both maps equally, briefly favoured the VAE map, concluded the VAE map is actively unhelpful and discarded it completely, and even with 100% seg uncertainty the result is still poor (Dice 0.08)
- α = 0.0 is the optimizer's verdict on the VAE image error map — it contributes nothing useful to localising adenomyosis features. This is consistent with everything seen throughout the project — the VAE reconstructs abnormal images too well to produce meaningful localised errors
- α drifted to 0.0 because the VAE image error map has no localisation value for adenomyosis detection — the optimizer correctly identified it as noise and discarded it entirely

---

## Per-Case Dice Scores

| Case | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| CC21460186Day of ETSDMODA2 | 0.0660 | 0.0876 | 0.1333 | 0.0545 |
| CM 41060268 SDFTET | 0.0416 | 0.0634 | 0.0629 | 0.0589 |
| GC41173292 D11FTET | 0.0451 | 0.0584 | 0.0499 | 0.0953 |
| JM40632979Day3stimMODAObs1ii | 0.1988 | 0.1093 | 0.0797 | 0.1114 |

**Clinical interpretation of per-case scores:**

- **JM40632979Day3stimMODAObs1ii** is the best performing case (Dice up to 0.1988 on s0) — this suggests the error maps for this case have slightly better spatial agreement with the annotated adenomyosis features, possibly because the pathology is more pronounced or the junctional zone disruption is more visible
- **CM 41060268 SDFTET** is the worst performing case (Dice as low as 0.0416) — the model's predictions are least aligned with the annotations here, suggesting the adenomyosis features in this case are subtle and not well captured by either error map
- The variation across slices within the same case (e.g. GC41173292 s0=0.0451 vs s3=0.0953) reflects that some slices show more pronounced features than others — consistent with adenomyosis being a 3D condition that varies in severity across the uterus
- All Dice scores are below 0.20 — clinically this means the system cannot yet reliably localise adenomyosis features to a level useful for clinical decision support
- A clinically useful system would need Dice > 0.50 as a minimum threshold for spatial agreement to be meaningful

---

## Prediction Visualisations

<!-- Prediction images and analysis will be added below as we review each case -->
