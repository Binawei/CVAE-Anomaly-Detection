"""
Visualization functions for Phase 2A results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from combine_maps import combine_error_maps_hard
import config


def plot_training_curves(gd_history, output_dir):
    """
    Plot training and validation loss curves from gradient descent.
    
    Args:
        gd_history: History dict from gradient descent
        output_dir: Where to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(gd_history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, gd_history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, gd_history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Dice Loss')
    axes[0, 0].set_title('Training Progress', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Parameter evolution: threshold_img
    axes[0, 1].plot(epochs, gd_history['threshold_img'], color='red', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Threshold Value')
    axes[0, 1].set_title('Image Error Threshold Evolution', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Parameter evolution: threshold_seg
    axes[1, 0].plot(epochs, gd_history['threshold_seg'], color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Threshold Value')
    axes[1, 0].set_title('Segmentation Uncertainty Threshold Evolution', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Parameter evolution: alpha
    axes[1, 1].plot(epochs, gd_history['alpha'], color='blue', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Alpha Value')
    axes[1, 1].set_title('Weighting Parameter (α) Evolution', fontweight='bold')
    axes[1, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal weight')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: training_curves.png")


def visualize_prediction(data, params, method_name, output_dir):
    """
    Visualize prediction for one case.
    
    Args:
        data: Data dict with error maps and annotation
        params: Parameter dict
        method_name: 'gradient_descent' or 'grid_search'
        output_dir: Where to save
    """
    # Generate prediction
    predicted = combine_error_maps_hard(
        data['img_error'],
        data['seg_uncertainty'],
        torch.tensor(params['threshold_img']),
        torch.tensor(params['threshold_seg']),
        torch.tensor(params['alpha'])
    )
    
    # Convert to numpy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()
    if isinstance(data['annotation'], torch.Tensor):
        annotation = data['annotation'].cpu().numpy()
    else:
        annotation = data['annotation']
    if isinstance(data['img_error'], torch.Tensor):
        img_err = data['img_error'].cpu().numpy()
        seg_unc = data['seg_uncertainty'].cpu().numpy()
    else:
        img_err = data['img_error']
        seg_unc = data['seg_uncertainty']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Error maps and ground truth
    axes[0, 0].imshow(img_err, cmap='hot', vmin=0, vmax=0.5)
    axes[0, 0].set_title('Image Reconstruction Error', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0], fraction=0.046)
    
    axes[0, 1].imshow(seg_unc, cmap='hot', vmin=0, vmax=0.5)
    axes[0, 1].set_title('Segmentation Uncertainty', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
    
    axes[0, 2].imshow(annotation, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth\n(Manual Annotation)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Prediction, comparison, overlay
    axes[1, 0].imshow(predicted, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Prediction ({method_name})', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Comparison (green=correct, red=error)
    comparison = np.zeros((*predicted.shape, 3))
    comparison[..., 1] = np.logical_and(predicted > 0.5, annotation > 0.5)  # True positives (green)
    comparison[..., 0] = np.logical_xor(predicted > 0.5, annotation > 0.5)  # Errors (red)
    axes[1, 1].imshow(comparison)
    axes[1, 1].set_title('Comparison\n🟢 Correct 🔴 Error', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = np.stack([img_err/img_err.max()]*3, axis=-1)
    overlay[annotation > 0.5, 1] = 1  # Ground truth in green
    overlay[predicted > 0.5, 0] = 1   # Prediction in red
    # Overlap appears yellow
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Overlay\n🟢 GT 🔴 Pred 🟡 Overlap', fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add parameters as text
    param_text = f"Parameters:\n"
    param_text += f"  θ_img = {params['threshold_img']:.3f}\n"
    param_text += f"  θ_seg = {params['threshold_seg']:.3f}\n"
    param_text += f"  α = {params['alpha']:.3f}"
    fig.text(0.98, 0.02, param_text, ha='right', va='bottom', 
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"{data['case_name']} - {method_name.replace('_', ' ').title()}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f"{data['case_name'].replace(' ', '_')}_{method_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def visualize_all_cases(data_list, params, method_name, output_dir):
    """
    Create visualizations for all cases.
    
    Args:
        data_list: List of data dicts
        params: Parameters to use
        method_name: Method name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations for {method_name}...")
    for data in data_list:
        visualize_prediction(data, params, method_name, output_dir)
        print(f"  ✓ {data['case_name']}")


def compare_methods(gd_params, gs_params, val_data, output_dir):
    """
    Create comparison between gradient descent and grid search.
    
    Args:
        gd_params: Gradient descent best params
        gs_params: Grid search best params
        val_data: Validation data
        output_dir: Output directory
    """
    from metrics import compute_all_metrics
    
    # Compute metrics for both methods on validation set
    gd_metrics = []
    gs_metrics = []
    
    for data in val_data:
        # Gradient descent prediction
        gd_pred = combine_error_maps_hard(
            data['img_error'], data['seg_uncertainty'],
            torch.tensor(gd_params['threshold_img']),
            torch.tensor(gd_params['threshold_seg']),
            torch.tensor(gd_params['alpha'])
        )
        gd_metrics.append(compute_all_metrics(gd_pred, data['annotation']))
        
        # Grid search prediction
        gs_pred = combine_error_maps_hard(
            data['img_error'], data['seg_uncertainty'],
            torch.tensor(gs_params['threshold_img']),
            torch.tensor(gs_params['threshold_seg']),
            torch.tensor(gs_params['alpha'])
        )
        gs_metrics.append(compute_all_metrics(gs_pred, data['annotation']))
    
    # Average metrics
    from metrics import average_metrics
    gd_avg = average_metrics(gd_metrics)
    gs_avg = average_metrics(gs_metrics)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    gd_values = [gd_avg[m] for m in metrics_to_plot]
    gs_values = [gs_avg[m] for m in metrics_to_plot]
    
    ax.bar(x - width/2, gd_values, width, label='Gradient Descent', alpha=0.8)
    ax.bar(x + width/2, gs_values, width, label='Grid Search', alpha=0.8)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Method Comparison on Validation Set', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved: method_comparison.png")
    
    return gd_avg, gs_avg


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
