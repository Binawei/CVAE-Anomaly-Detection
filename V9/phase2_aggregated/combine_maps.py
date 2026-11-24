"""
Functions to combine error maps using logical AND.
"""

import torch
from soft_threshold import soft_threshold, hard_threshold


def combine_error_maps_soft(img_error, seg_uncertainty,
                            threshold_img, threshold_seg,
                            steepness=50):
    """
    Combine error maps using soft logical AND (differentiable).

    Formula:
        combined = soft_threshold(img_error, θ_img) × soft_threshold(seg_uncertainty, θ_seg)
    """
    mask_img = soft_threshold(img_error, threshold_img, steepness)
    mask_seg = soft_threshold(seg_uncertainty, threshold_seg, steepness)
    return mask_img * mask_seg  # element-wise multiply = soft AND


def combine_error_maps_hard(img_error, seg_uncertainty,
                            threshold_img, threshold_seg):
    """
    Combine error maps using hard logical AND (for final evaluation).

    Formula:
        combined = (img_error > θ_img) AND (seg_uncertainty > θ_seg)
    """
    mask_img = hard_threshold(img_error, threshold_img)
    mask_seg = hard_threshold(seg_uncertainty, threshold_seg)
    return mask_img * mask_seg  # element-wise multiply = hard AND
