"""
Functions to combine error maps with learned parameters.
"""

import torch
from soft_threshold import soft_threshold, hard_threshold


def combine_error_maps_soft(img_error, seg_uncertainty, 
                            threshold_img, threshold_seg, alpha, 
                            steepness=10):
    """
    Combine error maps using soft thresholding (differentiable).
    
    Args:
        img_error: Image reconstruction error map (H, W)
        seg_uncertainty: Segmentation uncertainty map (H, W)
        threshold_img: Threshold for image error
        threshold_seg: Threshold for seg uncertainty
        alpha: Combination weight
        steepness: Soft threshold steepness
    
    Returns:
        Combined abnormality map (H, W) with values in [0, 1]
        
    Formula:
        combined = α × soft_threshold(img_error, threshold_img) + 
                   (1-α) × soft_threshold(seg_uncertainty, threshold_seg)
    """
    # Apply soft thresholding
    mask_img = soft_threshold(img_error, threshold_img, steepness)
    mask_seg = soft_threshold(seg_uncertainty, threshold_seg, steepness)
    
    # Weighted combination
    combined = alpha * mask_img + (1 - alpha) * mask_seg

    #use logical AND to combine this
    
    return combined


def combine_error_maps_hard(img_error, seg_uncertainty,
                            threshold_img, threshold_seg, alpha):
    """
    Combine error maps using hard thresholding (for final evaluation).
    
    Args:
        img_error: Image reconstruction error map (H, W)
        seg_uncertainty: Segmentation uncertainty map (H, W)
        threshold_img: Threshold for image error
        threshold_seg: Threshold for seg uncertainty
        alpha: Combination weight
    
    Returns:
        Combined abnormality map (H, W) with values in [0, 1]
    """
    # Apply hard thresholding
    mask_img = hard_threshold(img_error, threshold_img)
    mask_seg = hard_threshold(seg_uncertainty, threshold_seg)
    
    # Weighted combination
    combined = alpha * mask_img + (1 - alpha) * mask_seg
    
    return combined


def apply_to_batch(batch_data, threshold_img, threshold_seg, alpha, use_soft=True):
    """
    Apply combination to a batch of cases.
    
    Args:
        batch_data: List of data dicts with 'img_error', 'seg_uncertainty'
        threshold_img, threshold_seg, alpha: Parameters
        use_soft: Use soft (True) or hard (False) thresholding
    
    Returns:
        List of combined maps
    """
    combine_fn = combine_error_maps_soft if use_soft else combine_error_maps_hard
    
    results = []
    for data in batch_data:
        combined = combine_fn(
            data['img_error'],
            data['seg_uncertainty'],
            threshold_img,
            threshold_seg,
            alpha
        )
        results.append(combined)
    
    return results


if __name__ == "__main__":
    # Test combination
    import numpy as np
    
    # Create dummy data
    img_err = torch.tensor(np.random.rand(256, 256).astype(np.float32) * 0.3)
    seg_unc = torch.tensor(np.random.rand(256, 256).astype(np.float32) * 0.4)
    
    # Test parameters
    threshold_img = torch.tensor(0.15)
    threshold_seg = torch.tensor(0.25)
    alpha = torch.tensor(0.5)
    
    # Combine
    combined_soft = combine_error_maps_soft(img_err, seg_unc, 
                                           threshold_img, threshold_seg, alpha)
    combined_hard = combine_error_maps_hard(img_err, seg_unc,
                                           threshold_img, threshold_seg, alpha)
    
    print("Soft combination:")
    print(f"  Range: [{combined_soft.min():.3f}, {combined_soft.max():.3f}]")
    print(f"  Mean: {combined_soft.mean():.3f}")
    
    print("\nHard combination:")
    print(f"  Range: [{combined_hard.min():.3f}, {combined_hard.max():.3f}]")
    print(f"  Mean: {combined_hard.mean():.3f}")
