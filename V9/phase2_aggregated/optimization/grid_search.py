"""
Grid Search optimization for comparison with gradient descent.
Tests all combinations of parameter values.
"""

import torch
import numpy as np
import os
import sys
from itertools import product
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from combine_maps import combine_error_maps_hard
from losses import dice_loss
from metrics import compute_all_metrics


def evaluate_params(train_data, params):
    """
    Evaluate a set of parameters on training data.
    
    Args:
        train_data: List of training cases
        params: Dict with 'threshold_img', 'threshold_seg', 'alpha'
    
    Returns:
        Average Dice loss
    """
    total_loss = 0
    
    for data in train_data:
        # Combine with hard thresholding (no gradients needed)
        combined = combine_error_maps_hard(
            data['img_error'],
            data['seg_uncertainty'],
            torch.tensor(params['threshold_img']),
            torch.tensor(params['threshold_seg']),
            torch.tensor(params['alpha'])
        )
        
        # Compute loss
        loss = dice_loss(combined, data['annotation'])
        total_loss += loss.item()
    
    return total_loss / len(train_data)


def grid_search(train_data, val_data):
    """
    Perform grid search over parameter space.
    
    Args:
        train_data: Training cases
        val_data: Validation cases
    
    Returns:
        Tuple of (best_params, all_results)
    """
    print("\n" + "="*80)
    print("GRID SEARCH OPTIMIZATION")
    print("="*80)
    
    # Generate all parameter combinations
    param_combinations = list(product(
        config.GRID_SEARCH_RANGES['threshold_img'],
        config.GRID_SEARCH_RANGES['threshold_seg'],
        config.GRID_SEARCH_RANGES['alpha']
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations:")
    print(f"  threshold_img: {config.GRID_SEARCH_RANGES['threshold_img']}")
    print(f"  threshold_seg: {config.GRID_SEARCH_RANGES['threshold_seg']}")
    print(f"  alpha: {config.GRID_SEARCH_RANGES['alpha']}")
    print("="*80)
    
    # Evaluate all combinations
    results = []
    best_train_loss = float('inf')
    best_params = None
    
    for threshold_img, threshold_seg, alpha in tqdm(param_combinations, desc="Grid Search"):
        params = {
            'threshold_img': threshold_img,
            'threshold_seg': threshold_seg,
            'alpha': alpha
        }
        
        # Evaluate on training set
        train_loss = evaluate_params(train_data, params)
        
        # Store result
        results.append({
            'params': params.copy(),
            'train_loss': train_loss
        })
        
        # Track best
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_params = params.copy()
    
    # Evaluate best params on validation set
    val_loss = evaluate_params(val_data, best_params)
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Best training loss: {best_train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Best parameters:")
    print(f"  threshold_img: {best_params['threshold_img']:.4f}")
    print(f"  threshold_seg: {best_params['threshold_seg']:.4f}")
    print(f"  alpha: {best_params['alpha']:.4f}")
    print("="*80)
    
    # Add validation loss to best params
    best_params['train_loss'] = best_train_loss
    best_params['val_loss'] = val_loss
    
    return best_params, results


def run_grid_search(train_data, val_data):
    """
    Convenience function to run grid search.
    
    Args:
        train_data: Training cases
        val_data: Validation cases
    
    Returns:
        Tuple of (best_params, all_results)
    """
    return grid_search(train_data, val_data)


if __name__ == "__main__":
    # Test with dummy data
    print("Testing grid search optimizer...")
    
    # Create dummy data
    dummy_train = [
        {
            'case_name': f'train_{i}',
            'img_error': torch.rand(256, 256),
            'seg_uncertainty': torch.rand(256, 256),
            'annotation': (torch.rand(256, 256) > 0.7).float()
        }
        for i in range(4)
    ]
    
    dummy_val = [
        {
            'case_name': f'val_{i}',
            'img_error': torch.rand(256, 256),
            'seg_uncertainty': torch.rand(256, 256),
            'annotation': (torch.rand(256, 256) > 0.7).float()
        }
        for i in range(2)
    ]
    
    best_params, results = run_grid_search(dummy_train, dummy_val)
    print(f"\nTested {len(results)} combinations")
    print("Test completed successfully!")
