"""
Quick test to verify Phase 2A setup is correct.
Run this before running the full pipeline.
"""

import os
import sys

print("="*80)
print("PHASE 2A SETUP VERIFICATION")
print("="*80)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    import config
    print("  ✓ config.py")
    
    import data_loader
    print("  ✓ data_loader.py")
    
    import soft_threshold
    print("  ✓ soft_threshold.py")
    
    import combine_maps
    print("  ✓ combine_maps.py")
    
    import losses
    print("  ✓ losses.py")
    
    import metrics
    print("  ✓ metrics.py")
    
    from optimization import gradient_descent
    print("  ✓ optimization/gradient_descent.py")
    
    from optimization import grid_search
    print("  ✓ optimization/grid_search.py")
    
    import visualize
    print("  ✓ visualize.py")
    
    print("\n✓ All modules imported successfully!")
    
except Exception as e:
    print(f"\n✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check paths
print("\n2. Checking paths...")

print(f"  Annotations dir: {config.ANNOTATIONS_DIR}")
if os.path.exists(config.ANNOTATIONS_DIR):
    print(f"    ✓ Exists")
    files = os.listdir(config.ANNOTATIONS_DIR)
    print(f"    Found {len(files)} files")
else:
    print(f"    ✗ Does not exist!")

print(f"  Error maps dir: {config.ERROR_MAPS_DIR}")
if os.path.exists(config.ERROR_MAPS_DIR):
    print(f"    ✓ Exists")
else:
    print(f"    ✗ Does not exist!")

print(f"  Results dir: {config.RESULTS_DIR}")
print(f"    Will be created: {config.RESULTS_DIR}")

# Test 3: Check annotation files
print("\n3. Checking annotation files...")
missing = []
for case_name in config.get_all_cases():
    try:
        ann_path = config.get_annotation_path(case_name)
        if os.path.exists(ann_path):
            print(f"  ✓ {case_name}")
        else:
            print(f"  ✗ {case_name} - File not found: {ann_path}")
            missing.append(case_name)
    except Exception as e:
        print(f"  ✗ {case_name} - Error: {e}")
        missing.append(case_name)

if missing:
    print(f"\n⚠ Missing {len(missing)} annotation files!")
    print("  Please check file names in config.py SEGMENTATION_FILES")
else:
    print(f"\n✓ All {len(config.get_all_cases())} annotation files found!")

# Test 4: Check error maps for one case
print("\n4. Checking error maps (sample)...")
test_case = config.TRAIN_CASES[0]
print(f"  Testing with: {test_case}")

for slice_num in range(4):
    try:
        img_err_path = config.get_error_map_path(test_case, 'image_errors', slice_num)
        seg_err_path = config.get_error_map_path(test_case, 'seg_errors_aggregated', slice_num)
        
        if os.path.exists(img_err_path) and os.path.exists(seg_err_path):
            print(f"  ✓ Slice {slice_num}: Both error maps exist")
            break
        elif slice_num == 3:
            print(f"  ✗ No error maps found for any slice")
    except Exception as e:
        print(f"  ✗ Error checking slice {slice_num}: {e}")

# Test 5: Try loading one case
print("\n5. Testing data loading...")
try:
    from data_loader import load_annotation_volume, load_error_maps
    
    test_case = config.TRAIN_CASES[0]
    print(f"  Loading: {test_case}")
    
    # Load annotation
    annotation_slices = load_annotation_volume(test_case)
    print(f"  ✓ Annotation slices: {len(annotation_slices)}")
    if len(annotation_slices) > 0:
        print(f"    First slice shape: {annotation_slices[0].shape}")
        print(f"    Annotated pixels: {annotation_slices[0].sum():.0f}")
    
    # Load error maps
    for slice_num in range(4):
        try:
            img_err, seg_unc = load_error_maps(test_case, slice_num)
            print(f"  ✓ Slice {slice_num} error maps: {img_err.shape}, {seg_unc.shape}")
            print(f"    Image error range: [{img_err.min():.3f}, {img_err.max():.3f}]")
            print(f"    Seg uncertainty range: [{seg_unc.min():.3f}, {seg_unc.max():.3f}]")
            break
        except FileNotFoundError:
            continue
    
    print("\n✓ Data loading successful!")
    
except Exception as e:
    print(f"\n✗ Data loading error: {e}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "="*80)
print("SETUP VERIFICATION COMPLETE")
print("="*80)
print("\nIf all tests passed, you can run:")
print("  python run_phase2a.py")
print("\nIf any tests failed, fix the issues first.")
print("="*80 + "\n")
