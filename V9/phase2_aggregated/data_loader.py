"""
Data loader for Phase 2A.
Loads manual annotations and error maps for training and validation.
"""

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
import os
import config


def load_annotation_volume(case_name, target_size=(256, 256)):
    """
    Load ALL slices from manual annotation segmentation for a case.

    Args:
        case_name: Volume name
        target_size: Resize each slice to this size (256, 256) to match error maps
    
    Returns:
        List of binary masks, one per slice. Each is numpy array (H, W) with values 0 and 1
    """
    ann_path = config.get_annotation_path(case_name)
    
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation not found: {ann_path}")
    
    # Load NIfTI
    nii_img = nib.load(ann_path)
    data = nii_img.get_fdata()
    
    # Handle 4D data - take first volume
    if data.ndim > 3:
        data = data[:, :, :, 0]
    
    # Ensure 3D (or convert 2D to 3D with 1 slice)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]  # Add slice dimension
    
    # Process each slice
    slices = []
    for slice_idx in range(data.shape[2]):
        slice_data = data[:, :, slice_idx].squeeze()
        
        # Binarize (in case values are not exactly 0 and 1)
        slice_data = (slice_data > 0.5).astype(np.float32)
        
        # Resize to match error map size (256×256)
        if slice_data.shape != target_size:
            zoom_factors = (target_size[0] / slice_data.shape[0], 
                           target_size[1] / slice_data.shape[1])
            slice_data = zoom(slice_data, zoom_factors, order=0)  # Nearest neighbor for binary mask
        
        # Apply rotation and flip to match 3D Slicer orientation
        slice_data = np.fliplr(np.rot90(slice_data, k=1))
        
        # Make contiguous copy to fix negative stride issue
        slice_data = np.ascontiguousarray(slice_data)
        
        slices.append(slice_data)
    
    return slices


def load_error_maps(case_name, slice_num=0):
    """
    Load image error and segmentation uncertainty maps for a case.
    
    Args:
        case_name: Volume name
        slice_num: Which slice to load (0-3)
    
    Returns:
        Tuple of (image_error, seg_uncertainty) as numpy arrays (256, 256)
    """
    # Load image reconstruction error
    img_err_path = config.get_error_map_path(case_name, 'image_errors', slice_num)
    if not os.path.exists(img_err_path):
        raise FileNotFoundError(f"Image error map not found: {img_err_path}")
    img_err = np.load(img_err_path).astype(np.float32)
    img_err = np.ascontiguousarray(np.fliplr(np.rot90(img_err, k=1)))

    # Load aggregated segmentation uncertainty
    seg_unc_path = config.get_error_map_path(case_name, 'seg_errors_aggregated', slice_num)
    if not os.path.exists(seg_unc_path):
        raise FileNotFoundError(f"Seg uncertainty map not found: {seg_unc_path}")
    seg_unc = np.load(seg_unc_path).astype(np.float32)
    seg_unc = np.ascontiguousarray(np.fliplr(np.rot90(seg_unc, k=1)))

    return img_err, seg_unc


def prepare_training_data():
    """
    Load all training cases with ALL slices.

    Returns:
        List of dicts, each containing:
            - 'case_name': str (with slice number appended)
            - 'slice_num': int
            - 'annotation': np.array (256, 256)
            - 'img_error': np.array (256, 256)
            - 'seg_uncertainty': np.array (256, 256)
    """
    print(f"\nLoading {len(config.TRAIN_CASES)} training cases (all slices)...")

    data = []
    total_slices = 0

    for case_name in config.TRAIN_CASES:
        try:
            # Load ALL annotation slices
            annotation_slices = load_annotation_volume(case_name)
            num_slices = len(annotation_slices)

            print(f"\n  {case_name}: {num_slices} slices")

            # Match each annotation slice to corresponding error map
            for slice_idx in range(num_slices):  # Use ALL slices available
                try:
                    # Load error maps for this slice
                    img_err, seg_unc = load_error_maps(case_name, slice_idx)
                    annotation = annotation_slices[slice_idx]

                    # Count annotated pixels
                    ann_pixels = annotation.sum()

                    # Add to dataset (even if annotation is sparse - you annotated it!)
                    data.append({
                        'case_name': f"{case_name}_s{slice_idx}",
                        'slice_num': slice_idx,
                        'annotation': annotation,
                        'img_error': img_err,
                        'seg_uncertainty': seg_unc
                    })

                    total_slices += 1

                    print(f"    ✓ Slice {slice_idx}: {ann_pixels:.0f} annotated pixels | "
                          f"img_err=[{img_err.min():.3f}, {img_err.max():.3f}] | "
                          f"seg_unc=[{seg_unc.min():.3f}, {seg_unc.max():.3f}]")

                except FileNotFoundError as e:
                    print(f"    ✗ Slice {slice_idx}: Error map not found")
                    continue
                except Exception as e:
                    print(f"    ✗ Slice {slice_idx}: {e}")
                    continue

        except Exception as e:
            print(f"  ✗ Failed to load {case_name}: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"✓ Loaded {total_slices} slices from {len(config.TRAIN_CASES)} training cases")
    if len(config.TRAIN_CASES) > 0:
        print(f"  Average: {total_slices/len(config.TRAIN_CASES):.1f} slices per case")
    print(f"{'='*80}")
    return data


def prepare_validation_data():
    """
    Load all validation cases with ALL slices.

    Returns:
        List of dicts (same format as training data)
    """
    print(f"\nLoading {len(config.VAL_CASES)} validation cases (all slices)...")

    data = []
    total_slices = 0

    for case_name in config.VAL_CASES:
        try:
            # Load ALL annotation slices
            annotation_slices = load_annotation_volume(case_name)
            num_slices = len(annotation_slices)

            print(f"\n  {case_name}: {num_slices} slices")

            # Match each annotation slice to corresponding error map
            for slice_idx in range(num_slices):  # Use ALL slices available
                try:
                    img_err, seg_unc = load_error_maps(case_name, slice_idx)
                    annotation = annotation_slices[slice_idx]

                    ann_pixels = annotation.sum()

                    data.append({
                        'case_name': f"{case_name}_s{slice_idx}",
                        'slice_num': slice_idx,
                        'annotation': annotation,
                        'img_error': img_err,
                        'seg_uncertainty': seg_unc
                    })

                    total_slices += 1

                    print(f"    ✓ Slice {slice_idx}: {ann_pixels:.0f} annotated pixels")

                except FileNotFoundError:
                    print(f"    ✗ Slice {slice_idx}: Error map not found")
                    continue
                except Exception as e:
                    print(f"    ✗ Slice {slice_idx}: {e}")
                    continue

        except Exception as e:
            print(f"  ✗ Failed to load {case_name}: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"✓ Loaded {total_slices} slices from {len(config.VAL_CASES)} validation cases")
    if len(config.VAL_CASES) > 0:
        print(f"  Average: {total_slices/len(config.VAL_CASES):.1f} slices per case")
    print(f"{'='*80}")
    return data


def compute_error_map_bounds(data_list):
    """
    Compute min and max of error maps across all cases.
    Used to set parameter bounds for optimisation.
    """
    img_min, img_max = float('inf'), float('-inf')
    seg_min, seg_max = float('inf'), float('-inf')

    for d in data_list:
        img_min = min(img_min, d['img_error'].min())
        img_max = max(img_max, d['img_error'].max())
        seg_min = min(seg_min, d['seg_uncertainty'].min())
        seg_max = max(seg_max, d['seg_uncertainty'].max())

    print(f"\nError map bounds (from training data):")
    print(f"  img_error:       [{img_min:.4f}, {img_max:.4f}]")
    print(f"  seg_uncertainty: [{seg_min:.4f}, {seg_max:.4f}]")

    return {
        'threshold_img': (float(img_min), float(img_max)),
        'threshold_seg': (float(seg_min), float(seg_max)),
    }


def to_tensor(data_dict):
    """Convert numpy arrays in data dict to PyTorch tensors."""
    return {
        'case_name': data_dict['case_name'],
        'slice_num': data_dict['slice_num'],
        'annotation': torch.from_numpy(data_dict['annotation']).to(config.DEVICE),
        'img_error': torch.from_numpy(data_dict['img_error']).to(config.DEVICE),
        'seg_uncertainty': torch.from_numpy(data_dict['seg_uncertainty']).to(config.DEVICE)
    }


if __name__ == "__main__":
    # Test data loading
    print("="*80)
    print("TESTING DATA LOADER")
    print("="*80)

    train_data = prepare_training_data()
    val_data = prepare_validation_data()

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Training cases loaded: {len(train_data)}")
    print(f"  Validation cases loaded: {len(val_data)}")
    print(f"{'='*80}")
