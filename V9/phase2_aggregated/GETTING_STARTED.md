# Getting Started with Phase 2A

## Quick Start Guide

### Step 1: Verify Setup (2 minutes)

```bash
cd /mnt/c/Users/Binawei\ Gbamila/Desktop/phd/projects/CVAE/V9/phase2_aggregated
python test_setup.py
```

**This checks:**
- All Python files import correctly
- Annotation files are found
- Error maps exist
- Data loads properly

**Expected output:** All ✓ checks pass

---

### Step 2: Run Phase 2A (~10-15 minutes)

```bash
python run_phase2a.py
```

**What happens:**
1. Loads 4 training + 4 validation cases (~1 min)
2. Runs gradient descent optimization (~5-8 min)
3. Runs grid search comparison (~2-3 min)
4. Evaluates both methods (~1 min)
5. Generates visualizations (~2 min)
6. Saves all results

---

### Step 3: Review Results

**Check the `results/` folder:**

```
results/
├── training_curves.png          ← Loss and parameter evolution
├── gradient_descent/
│   ├── best_params.json         ← Learned parameters
│   ├── metrics.json             ← Performance metrics
│   └── predictions/             ← Visual predictions
│       ├── case1_gradient_descent.png
│       ├── case2_gradient_descent.png
│       └── ...
├── grid_search/
│   ├── best_params.json
│   ├── metrics.json
│   └── predictions/
└── comparison/
    └── method_comparison.png    ← GD vs Grid Search
```

---

## Understanding the Output

### Terminal Output

You'll see sections like:

```
================================================================================
GRADIENT DESCENT OPTIMIZATION
================================================================================
Initial parameters:
  threshold_img: 0.1000
  threshold_seg: 0.2500
  alpha: 0.5000

Epoch  10/200 | Train Loss: 0.3421 | Val Loss: 0.3892 | θ_img: 0.112 | θ_seg: 0.241 | α: 0.523
  → New best! Val loss: 0.3892
Epoch  20/200 | Train Loss: 0.2987 | Val Loss: 0.3456 | θ_img: 0.128 | θ_seg: 0.228 | α: 0.601
  → New best! Val loss: 0.3456
...

Early stopping at epoch 67
No improvement for 20 epochs

================================================================================
OPTIMIZATION COMPLETE
================================================================================
Best validation loss: 0.3124
Best parameters:
  threshold_img: 0.1345
  threshold_seg: 0.2189
  alpha: 0.6234
```

### Best Parameters Interpretation

If you get:
```json
{
  "threshold_img": 0.1345,
  "threshold_seg": 0.2189,
  "alpha": 0.6234
}
```

**This means:**
- Pixels with image error > 0.1345 are considered abnormal
- Pixels with seg uncertainty > 0.2189 are considered abnormal
- Final score = 62% image error + 38% seg uncertainty
- Higher `alpha` → image error matters more
- Lower `alpha` → segmentation uncertainty matters more

### Metrics Interpretation

```json
{
  "dice": 0.7234,
  "dice_std": 0.0892,
  "iou": 0.6543,
  "precision": 0.7891,
  "recall": 0.7012
}
```

**What's good:**
- **Dice > 0.7**: Good overlap with manual annotations
- **Dice > 0.8**: Excellent performance
- **Dice < 0.5**: Poor performance, need investigation

**Precision vs Recall tradeoff:**
- High precision, low recall: Conservative (marks only obvious cases)
- Low precision, high recall: Aggressive (marks more, some wrong)
- Balanced: Both around 0.7-0.8

---

## Common Issues

### Issue 1: All tests fail in test_setup.py

**Cause:** Not in correct directory or wrong Python environment

**Fix:**
```bash
# Make sure you're in the right place
cd /mnt/c/Users/Binawei\ Gbamila/Desktop/phd/projects/CVAE/V9/phase2_aggregated

# Activate your virtual environment
source ../../.venv_cvae/bin/activate  # Linux/Mac
# or
../../.venv_cvae/Scripts/Activate.ps1  # Windows PowerShell
```

### Issue 2: "No training data loaded"

**Cause:** Annotation files not found

**Fix:**
1. Check annotations exist:
   ```bash
   ls /mnt/c/Users/Binawei\ Gbamila/Desktop/phd/segmentations/segs/
   ```

2. Verify file names in `config.py` match exactly

3. Check `SEGMENTATION_FILES` dictionary in config.py

### Issue 3: "Error map not found"

**Cause:** Phase 1 incomplete or file naming mismatch

**Fix:**
1. Verify error maps exist:
   ```bash
   ls ../../error_maps/abnormal/image_errors/ | head -5
   ```

2. Check naming matches between annotation and error maps

### Issue 4: Poor performance (Dice < 0.3)

**Possible causes:**
1. Annotations are incorrect or incomplete
2. Error maps don't correlate with abnormalities
3. Initial parameter guesses are way off

**Debug steps:**
1. Look at visualizations - do predictions make sense?
2. Check if annotations cover the right regions
3. Try adjusting initial parameters in config.py:
   ```python
   INITIAL_PARAMS = {
       'threshold_img': 0.05,  # Try lower
       'threshold_seg': 0.30,  # Try higher
       'alpha': 0.5
   }
   ```

### Issue 5: CUDA out of memory

**Fix:** Force CPU mode in config.py:
```python
DEVICE = torch.device('cpu')
```

---

## What to Do After Success

### 1. Analyze Results
- Which method performed better? (Usually gradient descent)
- What do learned parameters tell you?
  - Is `alpha` close to 0, 0.5, or 1?
  - Are thresholds reasonable based on error map ranges?

### 2. Discuss with Supervisor
- Share the `results/` folder
- Show training curves and predictions
- Discuss parameter values

### 3. Apply to Full Dataset
- Use learned parameters on all 480 images
- Generate abnormality scores for everything
- Identify high-confidence abnormal cases

### 4. Proceed to Phase 2B (Optional)
- Try separate tissue uncertainties
- Compare with aggregated approach
- See if more parameters help

---

## Key Files Reference

| File | Purpose | When to Edit |
|------|---------|--------------|
| `config.py` | Settings, case lists, parameters | If changing cases or tuning |
| `run_phase2a.py` | Main pipeline | Rarely (it's complete) |
| `test_setup.py` | Verify setup | Never (just run it) |
| `data_loader.py` | Load data | Only if file format changes |
| `optimization/gradient_descent.py` | Main optimizer | If changing algorithm |
| `optimization/grid_search.py` | Comparison | If changing grid ranges |

---

## Questions to Answer

After running Phase 2A, answer these:

1. **Did gradient descent converge?**
   - Check training curves
   - Did loss decrease smoothly?

2. **Which method won?**
   - Gradient descent or grid search?
   - By how much? (Dice difference)

3. **What do parameters suggest?**
   - Is image error or seg uncertainty more important? (check `alpha`)
   - Are thresholds reasonable?

4. **Do predictions make clinical sense?**
   - Look at visualizations
   - Do marked regions show adenomyosis features?

5. **Ready for next step?**
   - Good Dice scores (>0.6)?
   - Consistent across validation cases?
   - Supervisor approval?

---

## Getting Help

**If stuck:**
1. Run `test_setup.py` to identify the issue
2. Check error messages carefully
3. Review this guide's "Common Issues" section
4. Check `README.md` for detailed documentation
5. Ask supervisor with specific error messages

**Before asking for help, provide:**
- Full error message (copy-paste terminal output)
- What you were trying to do
- What you expected vs what happened
- Output of `test_setup.py`

---

Good luck! 🚀
