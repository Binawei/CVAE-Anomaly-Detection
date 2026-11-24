# Phase 2A: Aggregated Uncertainty Approach

Learning optimal thresholds and weights for combining image reconstruction errors and segmentation uncertainties to detect adenomyosis.

## Overview

**Goal:** Learn 3 parameters to optimally combine error maps:
- `threshold_img`: Threshold for image reconstruction error
- `threshold_seg`: Threshold for aggregated segmentation uncertainty
- `alpha`: Weighting parameter (α×img + (1-α)×seg)

**Methods:**
1. **Gradient Descent** (Main) - Differentiable optimization using Adam
2. **Grid Search** (Comparison) - Exhaustive search over parameter grid

**Data:**
- Training: 4 manually annotated cases
- Validation: 4 manually annotated cases

---

## Quick Start

### 1. Check Configuration

```bash
python config.py
```

This shows:
- Training/validation case split
- Parameter ranges
- Paths to data

### 2. Test Data Loading

```bash
python data_loader.py
```

Verifies:
- Annotations load correctly
- Error maps exist
- Data shapes match

### 3. Run Complete Pipeline

```bash
python run_phase2a.py
```

This will:
1. Load 8 annotated cases
2. Run gradient descent optimization (~5-10 min)
3. Run grid search for comparison (~2-3 min)
4. Evaluate on validation set
5. Generate visualizations
6. Save all results

---

## File Structure

```
phase2_aggregated/
├── config.py                      # Configuration settings
├── data_loader.py                 # Load annotations + error maps
├── soft_threshold.py              # Differentiable thresholding
├── combine_maps.py                # Combine error maps
├── losses.py                      # Dice loss function
├── metrics.py                     # Evaluation metrics
│
├── optimization/
│   ├── gradient_descent.py        # Main optimization method
│   └── grid_search.py             # Comparison method
│
├── visualize.py                   # Plotting functions
├── run_phase2a.py                 # Master pipeline script
│
└── results/                       # Auto-generated outputs
    ├── gradient_descent/
    │   ├── best_params.json
    │   ├── history.npy
    │   ├── metrics.json
    │   └── predictions/           # PNG visualizations
    ├── grid_search/
    │   ├── best_params.json
    │   ├── all_results.json
    │   ├── metrics.json
    │   └── predictions/
    └── comparison/
        └── method_comparison.png
```

---

## Configuration

### Training Cases
```python
TRAIN_CASES = [
    "APU_FL3300 D6stim ADENO 22.3.21",
    "TL04051640D12stimAdeno",
    "KC41714618Day9stimAdenoexamplevieo",
    "CC21460186D10FTETSDMODAObst1i"
]
```

### Validation Cases
```python
VAL_CASES = [
    "CC21460186Day of ETSOMODA2",
    "CM 41060268 SDFTET",
    "GC41173292 D11FTET",
    "JM406329Day3stimMODAObs1ii"
]
```

### Gradient Descent Settings
```python
LEARNING_RATE = 0.01
MAX_EPOCHS = 200
PATIENCE = 20  # Early stopping
STEEPNESS = 10  # Soft threshold smoothness
```

### Grid Search Ranges
```python
GRID_SEARCH_RANGES = {
    'threshold_img': [0.05, 0.10, 0.15, 0.20],
    'threshold_seg': [0.15, 0.20, 0.25, 0.30],
    'alpha': [0.0, 0.33, 0.67, 1.0]
}
# Total: 64 combinations
```

---

## Understanding the Results

### Best Parameters

Example output:
```json
{
  "threshold_img": 0.1234,
  "threshold_seg": 0.2567,
  "alpha": 0.6789
}
```

**Interpretation:**
- `threshold_img = 0.12`: Pixels with image error > 0.12 are abnormal
- `threshold_seg = 0.26`: Pixels with seg uncertainty > 0.26 are abnormal
- `alpha = 0.68`: Image error gets 68% weight, seg uncertainty gets 32%

### Metrics

- **Dice Score** [0-1]: Overlap between prediction and ground truth (higher = better)
- **IoU** [0-1]: Intersection over union (similar to Dice)
- **Precision** [0-1]: Of pixels marked abnormal, how many were correct?
- **Recall** [0-1]: Of truly abnormal pixels, how many did we find?
- **F1 Score** [0-1]: Harmonic mean of precision and recall

### Visualizations

**training_curves.png**
- Loss over time
- Parameter evolution
- Shows convergence

**Prediction PNGs** (per case)
- Error maps (input)
- Ground truth annotation
- Model prediction
- Comparison overlay

**method_comparison.png**
- Gradient descent vs grid search
- Bar chart of all metrics

---

## Troubleshooting

### Error: "No training data loaded"
**Check:**
1. Annotation files exist in `C:\Users\Binawei Gbamila\Desktop\phd\segmentations\segs`
2. File names match `SEGMENTATION_FILES` in config.py
3. Error maps exist in `V9/error_maps/abnormal/`

### Error: "Shape mismatch"
**Solution:** Error maps are 256×256. Annotations are resized automatically, but check if loading correctly.

### Poor Performance (Dice < 0.3)
**Try:**
1. Adjust initial parameter values in config.py
2. Increase `MAX_EPOCHS` to 300-400
3. Decrease `LEARNING_RATE` to 0.005
4. Check if annotations are correct (visualize them)

### Gradient descent not converging
**Try:**
1. Lower learning rate (0.005 or 0.001)
2. Increase steepness (harder threshold, e.g., 20)
3. Add more training cases if available

---

## Next Steps

### After Phase 2A Success:
1. **Analyze results**: Which parameter matters most?
2. **Phase 2B**: Try separate tissue uncertainties instead of aggregated
3. **Apply to full dataset**: Use learned parameters on all 480 images
4. **Compare with clinician annotations**: When they review your labels

### If Results Are Poor:
1. **Check annotations**: Are they accurate?
2. **Visualize error maps**: Do they show abnormalities?
3. **Try different ranges**: Adjust grid search bounds
4. **Discuss with supervisor**: May need different approach

---

## Key Concepts

### Soft vs Hard Thresholding

**Hard** (non-differentiable):
```python
mask = (error > threshold)  # Step function
```

**Soft** (differentiable):
```python
mask = sigmoid(steepness × (error - threshold))  # Smooth approximation
```

Gradient descent uses soft during training, hard for final evaluation.

### Why Gradient Descent?

- **Continuous optimization**: Finds optimal values, not just grid points
- **Efficient**: Faster than exhaustive search
- **Scalable**: Works better when adding more parameters (Phase 2B)

### Why Grid Search for Comparison?

- **Sanity check**: Validates gradient descent results
- **Simple**: Easy to understand and debug
- **May find better**: Sometimes discrete search finds global optimum

---

## References

**Supervisor's Specifications:**
- Main method: Gradient descent
- Comparison: Small grid search (~4 values per parameter)
- Split: 4 training, 4 validation cases
- Start with aggregated, then try separate uncertainties

**Related Code:**
- Phase 1: `V9/step1_generate_image_errors.py` (image errors)
- Phase 1: `V9/step2_generate_seg_uncertainties.py` (seg uncertainty)
- Phase 1: `V9/step3_aggregate_seg_maps.py` (aggregation)

---

## Contact

For questions or issues, discuss with supervisor or check:
- Implementation plan: `V9/IMPLEMENTATION_PLAN.md`
- Main project README: `../README.md`
