"""
Phase 2A Master Script
Runs complete pipeline: data loading, gradient descent, grid search, evaluation.
"""

import os
import sys
import json
import torch
import numpy as np

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader import prepare_training_data, prepare_validation_data, to_tensor, compute_error_map_bounds
from optimization.gradient_descent import run_gradient_descent
from optimization.grid_search import run_grid_search
from metrics import compute_all_metrics, average_metrics, print_metrics
from visualize import (plot_training_curves, visualize_all_cases, 
                       compare_methods)


def save_results(gd_params, gd_history, gs_params, gs_results, 
                gd_metrics, gs_metrics, output_dir):
    """Save all results to files."""
    
    # Save gradient descent results
    gd_dir = os.path.join(output_dir, 'gradient_descent')
    os.makedirs(gd_dir, exist_ok=True)
    
    with open(os.path.join(gd_dir, 'best_params.json'), 'w') as f:
        json.dump(gd_params, f, indent=2)
    
    np.save(os.path.join(gd_dir, 'history.npy'), gd_history)
    
    with open(os.path.join(gd_dir, 'metrics.json'), 'w') as f:
        json.dump(gd_metrics, f, indent=2)
    
    # Save grid search results
    gs_dir = os.path.join(output_dir, 'grid_search')
    os.makedirs(gs_dir, exist_ok=True)
    
    with open(os.path.join(gs_dir, 'best_params.json'), 'w') as f:
        json.dump(gs_params, f, indent=2)
    
    with open(os.path.join(gs_dir, 'all_results.json'), 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for r in gs_results:
            serializable_results.append({
                'params': r['params'],
                'train_loss': float(r['train_loss'])
            })
        json.dump(serializable_results, f, indent=2)
    
    with open(os.path.join(gs_dir, 'metrics.json'), 'w') as f:
        json.dump(gs_metrics, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")


def main():
    """Main Phase 2A pipeline."""
    
    print("\n" + "="*80)
    print("PHASE 2A: AGGREGATED UNCERTAINTY APPROACH")
    print("Learning Thresholds and Weights for Abnormality Detection")
    print("="*80)
    
    # Print configuration
    config.print_config()
    
    # Step 1: Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    train_data = prepare_training_data()
    val_data = prepare_validation_data()
    
    if len(train_data) == 0:
        print("\n✗ ERROR: No training data loaded!")
        print("Please check:")
        print("  1. Annotation files exist in:", config.ANNOTATIONS_DIR)
        print("  2. Error maps exist in:", config.ERROR_MAPS_DIR)
        return
    
    if len(val_data) == 0:
        print("\n✗ ERROR: No validation data loaded!")
        return
    
    # Convert to tensors
    train_data_tensors = [to_tensor(d) for d in train_data]
    val_data_tensors = [to_tensor(d) for d in val_data]

    # Compute bounds from actual data
    bounds = compute_error_map_bounds(train_data_tensors)

    print(f"\n✓ Data loading complete")
    print(f"  Training cases: {len(train_data)}")
    print(f"  Validation cases: {len(val_data)}")
    
    # Step 2: Gradient Descent (Main Method)
    print("\n" + "="*80)
    print("STEP 2: GRADIENT DESCENT OPTIMIZATION (Main Method)")
    print("="*80)
    
    gd_params, gd_history = run_gradient_descent(train_data_tensors, val_data_tensors, bounds)

    # Step 3: Grid Search (Comparison) — disabled for now
    # gs_params, gs_results = run_grid_search(train_data_tensors, val_data_tensors)

    # Step 4: Evaluation on Validation Set
    print("\n" + "="*80)
    print("STEP 4: EVALUATION ON VALIDATION SET")
    print("="*80)

    from combine_maps import combine_error_maps_hard

    # Gradient descent evaluation
    print("\nGradient Descent:")
    gd_metrics_list = []
    for data in val_data_tensors:
        pred = combine_error_maps_hard(
            data['img_error'], data['seg_uncertainty'],
            torch.tensor(gd_params['threshold_img']),
            torch.tensor(gd_params['threshold_seg'])
        )
        metrics = compute_all_metrics(pred, data['annotation'])
        gd_metrics_list.append(metrics)
        print(f"  {data['case_name']}: Dice = {metrics['dice']:.4f}")

    gd_avg_metrics = average_metrics(gd_metrics_list)
    print_metrics(gd_avg_metrics, "Gradient Descent - Average")

    # Step 5: Visualization
    print("\n" + "="*80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*80)

    results_dir = config.RESULTS_DIR

    print("\nPlotting training curves...")
    plot_training_curves(gd_history, results_dir)

    gd_vis_dir = os.path.join(results_dir, 'gradient_descent', 'predictions')
    visualize_all_cases(val_data_tensors, gd_params, 'gradient_descent', gd_vis_dir)

    # Step 6: Save Results
    print("\n" + "="*80)
    print("STEP 6: SAVING RESULTS")
    print("="*80)

    gd_dir = os.path.join(results_dir, 'gradient_descent')
    os.makedirs(gd_dir, exist_ok=True)
    with open(os.path.join(gd_dir, 'best_params.json'), 'w') as f:
        json.dump(gd_params, f, indent=2)
    np.save(os.path.join(gd_dir, 'history.npy'), gd_history)
    with open(os.path.join(gd_dir, 'metrics.json'), 'w') as f:
        json.dump(gd_avg_metrics, f, indent=2)
    print(f"\n✓ Results saved to: {results_dir}")

    # Final Summary
    print("\n" + "="*80)
    print("PHASE 2A COMPLETE!")
    print("="*80)
    print("\nGradient Descent (Main Method):")
    print(f"  threshold_img: {gd_params['threshold_img']:.4f}")
    print(f"  threshold_seg: {gd_params['threshold_seg']:.4f}")
    print(f"  Validation Dice: {gd_avg_metrics['dice']:.4f} +/- {gd_avg_metrics['dice_std']:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
