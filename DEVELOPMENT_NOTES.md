# CVAE Anomaly Detection – Development Notes

## Project Overview

Unsupervised anomaly detection model using a Convolutional VAE (CVAE) trained exclusively on normal uterus ultrasound images. At inference, normal images produce low reconstruction error while abnormal (adenomyosis) images produce high reconstruction error due to the model never having seen them during training.

## Dataset
- 120 NIfTI files: 47 normal ("Norm" in filename), 73 abnormal
- Slices extracted from each NIfTI file, normalised, resized to 256×256, saved as PNG
- Total: 484 PNGs (≈190 normal, ≈294 abnormal)
- Training uses normal images only with 70/15/15 train/val/test split
- Effective training set: ≈133 normal images after split

## Architecture

5-layer encoder/decoder:
- Encoder: 256→128→64→32→16→8 spatial resolution
- Channels: 1→32→64→128→256→512
- Bottleneck: 1×1 Conv2d (512→LATENT_DIM), spatial 8×8
- Decoder mirrors encoder back to 256×256

---

## Version History

### V1 — Baseline
- **Architecture**: MaxPool2d downsampling + bilinear upsampling
- **Loss**: MSE + KL (β=0.1, warm-up over 20 epochs)
- **Config**: LATENT_DIM=16, BATCH_SIZE=16, EPOCHS=200
- **Best Val Loss**: 572.08
- **Result**: Very blurry reconstructions. LATENT_DIM=16 caused 64× compression (65,536→1,024), too aggressive for 256×256 images. Bilinear upsampling produced smooth outputs with no texture

### V2 — More Epochs
- **Architecture**: Same as V1
- **Loss**: MSE + KL (β=0.1)
- **Config**: LATENT_DIM=16, BATCH_SIZE=128, EPOCHS=600
- **Best Val Loss**: 462.8
- **Result**: Marginal improvement. Still blurry. BATCH_SIZE=128 identified as problematic for small dataset

### V3 — Architecture Overhaul
- **Architecture**: Replaced MaxPool2d with strided Conv2d (stride=2), replaced bilinear upsample with ConvTranspose2d
- **Loss**: MSE + SSIM + KL (β=0.1)
- **Config**: LATENT_DIM=64, BATCH_SIZE=16, EPOCHS=600
- **Best Val Loss**: 378.2
- **Result**: Noticeable improvement in texture. Strided convolutions learn what spatial information to retain during downsampling. ConvTranspose2d learns how to reconstruct detail during upsampling. SSIM penalises loss of edges and structure

### V4 — Lower KL Weight + Longer Training
- **Architecture**: Same as V3
- **Loss**: MSE + SSIM + KL (β=0.01, reduced from 0.1)
- **Config**: LATENT_DIM=64, BATCH_SIZE=16, EPOCHS=1000
- **Best Val Loss**: 197.7
- **Mean Normal Error**: 161.6 ± 31.1
- **Result**: Significant improvement. Reducing β from 0.1 to 0.01 reduced KL regularisation pressure, allowing the model to focus more on reconstruction quality. Rough tissue surfaces partially captured

### V5 — LATENT_DIM=128 (Failed)
- **Architecture**: Same as V3
- **Loss**: MSE + SSIM + KL (β=0.01)
- **Config**: LATENT_DIM=128, BATCH_SIZE=16, EPOCHS=1000
- **Best Val Loss**: 416.8
- **Result**: Worse than V4. With only ≈133 training images, the larger latent space (128×8×8=8,192 values) had too many parameters to learn effectively. Training instability observed (spike at epoch 940). LATENT_DIM=64 confirmed as the sweet spot for this dataset size

### V6 — Full Skip Connections (Rejected)
- **Architecture**: U-Net style skip connections at all 4 encoder levels
- **Loss**: MSE + SSIM + KL (β=0.01)
- **Config**: LATENT_DIM=64, BATCH_SIZE=16, EPOCHS=1000
- **Best Val Loss**: 4.4
- **Mean Normal Error**: 4.35 ± 0.38
- **Result**: Rejected for anomaly detection. Skip connections at all levels allowed the decoder to directly access fine-grained encoder features, effectively bypassing the bottleneck. The model could reconstruct any image — including abnormal ones — with near-perfect accuracy, making it unsuitable for anomaly detection. The combination of SSIM loss and full skip connections created a shortcut that made the latent space redundant

### V7 — Partial Skip Connection (Current Best)
- **Architecture**: Single skip connection at deepest encoder level only (enc[3]→dec[1], 256 channels, 16×16 spatial)
- **Loss**: MSE + SSIM + KL (β=0.01)
- **Config**: LATENT_DIM=64, BATCH_SIZE=16, EPOCHS=1000
- **Best Val Loss**: 51.4
- **Mean Normal Error**: 51.0 ± 7.8
- **Result**: Best balance achieved. Sharp reconstructions with tissue texture and rough surfaces well captured. Bottleneck integrity preserved — only the deepest semantic features are passed via skip connection, while fine-grained spatial details must still be reconstructed through the bottleneck. Tight error spread (±7.8) indicates consistent reconstruction across all normal test images

---

## Summary of Results

| Version | Key Change | Best Val Loss | Normal Error | Top-10% AUC |
|---------|-----------|---------------|--------------|-------------|
| V1 | Baseline (MaxPool + bilinear, MSE) | 572.1 | — | — |
| V2 | More epochs | 462.8 | — | — |
| V3 | Strided conv + ConvTranspose2d + SSIM | 378.2 | — | — |
| V4 | β reduced to 0.01, 1000 epochs | 197.7 | 161.6 ± 31.1 | — |
| V5 | LATENT_DIM=128 | 416.8 | — | — |
| V6 | Full skip connections (rejected) | 4.4 | 4.35 ± 0.38 | — |
| V7 | Partial skip connection | 51.4 | 51.0 ± 7.8 | — |
| V4 (rerun) | Pure VAE, balanced eval | 198.2 | 156.7 ± 34.9 | 0.436 |
| V8 | LATENT_DIM=32, β=0.1 | 363.2 | 260.5 ± 53.0 | 0.421 |

---

## Error Heatmap Analysis

### What was added
- `plot_error_heatmaps()` added to `evaluate.py` — groups slices by patient ID and plots `original | reconstructed | pixel-wise absolute error` for each slice
- `MetaImageDataset` added to `dataset.py` — returns `(image, patient_id, filepath)` tuples to enable patient-level grouping
- Patient ID extracted by stripping `_s{N}` suffix from filename

### Observations from heatmaps
- Error is concentrated along fine texture lines and edges (ultrasound speckle), not in anatomically meaningful regions
- Normal and abnormal heatmaps look virtually identical in character — the model cannot visually distinguish the two classes
- The model learned to reproduce the overall blob shape and brightness but ignores fine-grained tissue texture
- `CC21295280FreshDayofET` (abnormal): reconstruction is structurally wrong on some slices (decoder produces a different shape), yet scalar error is still lower than normal — confirms the mean MSE is dominated by large black background regions which reconstruct perfectly
- `ACV21021720` (normal): completely different image format (rectangular with heavy black padding) — model handles it fine, showing it is learning "put black where it is black" rather than normal-specific anatomy

### Root cause confirmed
The large black background regions reconstruct with near-zero error and drag the global mean down. Abnormal images tend to have more background or simpler tissue regions, so their global error is lower — not because the model detects them as normal, but because the background dominates the score.

---

## Scoring Improvements (No Retraining)

### Changes made to `evaluate.py`
- `_tissue_mask()` — binary mask of pixels above threshold=0.05, isolates tissue from background
- `_masked_error()` — computes two scores per image:
  - **Masked mean**: mean absolute error inside tissue region only
  - **Top-10% mean**: mean of the top 10% highest-error pixels inside tissue (captures localised anomalies)
- All three scores (global, masked, top-10%) reported with separate histograms and ROC curves

### Results — Balanced Evaluation (All Normal Images)

| Metric | Normal | Abnormal | AUC (15% split) | AUC (all normal) |
|--------|--------|----------|-----------------|------------------|
| Global (MSE+SSIM) | 156.72 ± 34.90 | 139.86 ± 31.28 | 0.293 | 0.339 |
| Masked Mean | 0.0413 ± 0.0035 | 0.0401 ± 0.0032 | 0.357 | 0.387 |
| Top-10% Mean | 0.1269 ± 0.0100 | 0.1265 ± 0.0120 | 0.405 | 0.436 |

### Additional finding — class imbalance in evaluation
- Original `get_test_loaders()` used only 15% normal test split (~30 images) vs all 196 abnormal images
- Added `get_test_loaders_all_normal()` to use all 204 normal images for evaluation
- Balancing alone improved every AUC metric by ~0.03-0.04
- Normal distribution is wider with a longer right tail than abnormal across all metrics
- Distributions overlap almost completely — not separable with current model
- **Conclusion**: post-processing improvements are exhausted. Retraining required

---

## Next Steps

---

## Texture Complexity Analysis

### Motivation
The persistent wrong-direction AUC (abnormal < normal reconstruction error) led to the hypothesis that adenomyosis ultrasound images may be structurally simpler than normal images. This was tested quantitatively on the dataset before making any claims.

### Method
`texture_analysis.py` computes three metrics inside the tissue region (pixels > 0.05 threshold) for every image:
- **Variance** — spread of pixel intensities, measures tonal complexity
- **Gradient Magnitude** — Sobel edge density, measures structural detail and texture
- **Shannon Entropy** — information content, measures overall image complexity

Statistical significance tested with Mann-Whitney U test (non-parametric, no distribution assumption).

### Results

| Metric | Normal Mean | Abnormal Mean | p-value | Direction |
|--------|-------------|---------------|---------|----------|
| Variance | 0.040081 | 0.035429 | < 0.0001 | Normal > Abnormal |
| Gradient Magnitude | 0.060740 | 0.057922 | < 0.0001 | Normal > Abnormal |
| Shannon Entropy | 6.385884 | 6.061629 | 0.0001 | Normal > Abnormal |

### Interpretation
- All three metrics are statistically significant (p < 0.05)
- All three point in the same direction — normal images are measurably more complex than abnormal images
- The VAE is behaving correctly — it reconstructs simpler images (abnormal) better than complex ones (normal)
- The reconstruction-based anomaly detection assumption breaks down when the anomaly class is structurally simpler than the normal class
- This is consistent with the clinical description of adenomyosis: loss of the normal junctional zone striation makes the myometrium more homogeneous
- **This is a dataset characteristic, not a model failure**

### Conclusion
The wrong-direction AUC is explained by the data, not the model. Reconstruction-based anomaly detection is fundamentally limited for this problem. The next step requires a discussion with the supervisor about whether labelled data can be used to guide the model.
