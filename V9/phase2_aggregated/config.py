"""
Phase 2A Configuration: Aggregated Uncertainty Approach
Learning thresholds and weights using gradient descent + grid search comparison.
"""

import os
import torch

# ============================================================================
# PATHS
# ============================================================================

# Project paths
V9_DIR = os.path.dirname(os.path.dirname(__file__))
CVAE_DIR = os.path.dirname(V9_DIR)

# Error maps location (generated in Phase 1)
ERROR_MAPS_DIR = os.path.join(V9_DIR, 'error_maps')

# Manual annotations location
# Support both Windows and WSL paths
import platform
if platform.system() == "Windows" or os.path.exists(r"C:\Users\Binawei Gbamila\Desktop\phd\segmentations\segs"):
    ANNOTATIONS_DIR = os.path.normpath(r"C:\Users\Binawei Gbamila\Desktop\phd\segmentations\segs")
else:
    # WSL/Linux path
    ANNOTATIONS_DIR = "/mnt/c/Users/Binawei Gbamila/Desktop/phd/segmentations/segs"

# Results output
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# LABELED CASES (8 Total: 4 Train, 4 Validation)
# ============================================================================

# Training set (used to learn parameters)
TRAIN_CASES = [
    "APU_FL3300 D6stim ADENO 22.3.21",
    "TL04051640D12stimAdeno",
    "KC41714618Day9stimAdenoexamplevideo",
    "CC21460186D10FTETSDMODAObs1i"
]

# Validation set (used to evaluate learned parameters)
VAL_CASES = [
    "CC21460186Day of ETSDMODA2",
    "CM 41060268 SDFTET",
    "GC41173292 D11FTET",
    "JM40632979Day3stimMODAObs1ii"
]

# Mapping from volume names to segmentation files
SEGMENTATION_FILES = {
    "APU_FL3300 D6stim ADENO 22.3.21": "APU_FL3300 D6stim ADENO 22.3.21-Segmentation.nii",
    "TL04051640D12stimAdeno": "TL04051640D12stimAdeno-Segmentation.nii",
    "KC41714618Day9stimAdenoexamplevideo": "KC41714618Day9stimAdenoexamplevideo-Segmentation.nii",
    "CC21460186D10FTETSDMODAObs1i": "CC21460186D10FTETSDMODAObs1i-Segmentation.nii",
    "CC21460186Day of ETSDMODA2": "CC21460186Day of ETSDMODA2-Segmentation.nii",
    "CM 41060268 SDFTET": "CM 41060268 SDFTET-Segmentation.nii",
    "GC41173292 D11FTET": "GC41173292 D11FTET-Segmentation.nii",
    "JM40632979Day3stimMODAObs1ii": "JM40632979Day3stimMODAObs1ii-Segmentation.nii"
}

# ============================================================================
# GRADIENT DESCENT SETTINGS (Main Optimization Method)
# ============================================================================

# Initial parameter values
INITIAL_PARAMS = {
    'threshold_img': 0.10,     # Image reconstruction error threshold
    'threshold_seg': 0.25,     # Segmentation uncertainty threshold (aggregated)
    'alpha': 0.5               # Combination weight: α×img + (1-α)×seg
}

# Parameter bounds (valid ranges)
PARAM_BOUNDS = {
    'threshold_img': (0.05, 0.50),
    'threshold_seg': (0.05, 0.50),
    'alpha': (0.0, 1.0) #Remove alpha and combine with logical AND.
}

# Optimization hyperparameters
LEARNING_RATE = 0.01
MAX_EPOCHS = 500
PATIENCE = 100
MIN_DELTA = 1e-5

STEEPNESS = 50

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ============================================================================
# GRID SEARCH SETTINGS (Comparison Method)
# ============================================================================

# Grid search ranges (~4 values per parameter as suggested by supervisor)
GRID_SEARCH_RANGES = {
    'threshold_img': [0.05, 0.10, 0.15, 0.20],
    'threshold_seg': [0.15, 0.20, 0.25, 0.30],
    'alpha': [0.0, 0.33, 0.67, 1.0]
}

# Total combinations: 4 × 4 × 4 = 64

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Image size (error maps are 256×256)
IMG_SIZE = 256

# Metrics to compute
METRICS = ['dice', 'iou', 'precision', 'recall', 'f1']

# Visualization settings
SAVE_PREDICTIONS = True  # Save predicted masks as images
SAVE_COMPARISONS = True  # Save side-by-side comparisons

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_cases():
    """Return all labeled cases (train + val)."""
    return TRAIN_CASES + VAL_CASES

def get_annotation_path(case_name):
    """Get full path to annotation file for a case."""
    seg_file = SEGMENTATION_FILES.get(case_name)
    if seg_file is None:
        raise ValueError(f"No segmentation file found for case: {case_name}")
    path = os.path.join(ANNOTATIONS_DIR, seg_file)
    return os.path.normpath(path)  # Normalize path separators for Windows

def get_error_map_path(case_name, error_type, slice_num=0):
    """
    Get path to error map file.
    
    Args:
        case_name: Volume name (e.g., "APU_FL3300 D6stim ADENO 22.3.21")
        error_type: 'image_errors' or 'seg_errors_aggregated'
        slice_num: Slice number (0, 1, 2, 3)
    """
    filename = f"{case_name}_s{slice_num}.npy"
    # All our labeled cases are abnormal
    path = os.path.join(ERROR_MAPS_DIR, 'abnormal', error_type, filename)
    return path

def print_config():
    """Print configuration summary."""
    print("="*80)
    print("PHASE 2A CONFIGURATION")
    print("="*80)
    print(f"\nTraining Cases: {len(TRAIN_CASES)}")
    for i, case in enumerate(TRAIN_CASES, 1):
        print(f"  {i}. {case}")
    
    print(f"\nValidation Cases: {len(VAL_CASES)}")
    for i, case in enumerate(VAL_CASES, 1):
        print(f"  {i}. {case}")
    
    print(f"\nGradient Descent Settings:")
    print(f"  Initial params: {INITIAL_PARAMS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  Device: {DEVICE}")
    
    print(f"\nGrid Search Settings:")
    print(f"  threshold_img: {GRID_SEARCH_RANGES['threshold_img']}")
    print(f"  threshold_seg: {GRID_SEARCH_RANGES['threshold_seg']}")
    print(f"  alpha: {GRID_SEARCH_RANGES['alpha']}")
    print(f"  Total combinations: {len(GRID_SEARCH_RANGES['threshold_img']) * len(GRID_SEARCH_RANGES['threshold_seg']) * len(GRID_SEARCH_RANGES['alpha'])}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_config()
