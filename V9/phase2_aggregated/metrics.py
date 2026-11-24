"""
Evaluation metrics for segmentation performance.
"""

import torch
import numpy as np


def dice_score(predicted, target, smooth=1e-6):
    """Dice similarity coefficient."""
    pred_flat = predicted.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    return (2 * intersection + smooth) / (union + smooth)


def iou_score(predicted, target, smooth=1e-6):
    """Intersection over Union (Jaccard index)."""
    pred_flat = predicted.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


def precision_score(predicted, target, smooth=1e-6):
    """Precision = TP / (TP + FP)."""
    pred_flat = predicted.view(-1)
    target_flat = target.view(-1)
    
    true_positive = (pred_flat * target_flat).sum()
    predicted_positive = pred_flat.sum()
    
    return (true_positive + smooth) / (predicted_positive + smooth)


def recall_score(predicted, target, smooth=1e-6):
    """Recall = TP / (TP + FN)."""
    pred_flat = predicted.view(-1)
    target_flat = target.view(-1)
    
    true_positive = (pred_flat * target_flat).sum()
    actual_positive = target_flat.sum()
    
    return (true_positive + smooth) / (actual_positive + smooth)


def f1_score(predicted, target):
    """F1 score = 2 × (precision × recall) / (precision + recall)."""
    prec = precision_score(predicted, target)
    rec = recall_score(predicted, target)
    
    return 2 * (prec * rec) / (prec + rec + 1e-6)


def compute_all_metrics(predicted, target):
    """
    Compute all metrics for a prediction.
    
    Args:
        predicted: Predicted mask tensor
        target: Ground truth mask tensor
    
    Returns:
        Dictionary with all metric values
    """
    # Ensure tensors
    if not isinstance(predicted, torch.Tensor):
        predicted = torch.from_numpy(predicted)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)
    
    # Binarize predicted (if not already)
    if predicted.max() > 1 or predicted.min() < 0:
        predicted = (predicted > 0.5).float()
    
    metrics = {
        'dice': dice_score(predicted, target).item(),
        'iou': iou_score(predicted, target).item(),
        'precision': precision_score(predicted, target).item(),
        'recall': recall_score(predicted, target).item(),
        'f1': f1_score(predicted, target).item()
    }
    
    return metrics


def average_metrics(metrics_list):
    """
    Average metrics across multiple cases.
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}
    
    keys = metrics_list[0].keys()
    averaged = {}
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        averaged[key] = np.mean(values)
        averaged[f'{key}_std'] = np.std(values)
    
    return averaged


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics."""
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in metrics.items():
        if not key.endswith('_std'):
            std_key = f'{key}_std'
            if std_key in metrics:
                print(f"  {key:12s}: {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {key:12s}: {value:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    # Test metrics
    pred = torch.rand(100, 100)
    target = (torch.rand(100, 100) > 0.5).float()
    
    metrics = compute_all_metrics(pred, target)
    print_metrics(metrics, "Test Metrics")
