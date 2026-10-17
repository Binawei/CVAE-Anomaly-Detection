"""
Loss functions for optimization.
"""

import torch


def dice_loss(predicted, target, smooth=1e-6):
    """
    Dice loss for binary segmentation.
    
    Args:
        predicted: Predicted mask (H, W) with values in [0, 1]
        target: Ground truth binary mask (H, W) with values 0 or 1
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss (scalar), where lower is better
        
    Formula:
        Dice = (2 × intersection + smooth) / (sum_pred + sum_target + smooth)
        Loss = 1 - Dice
    """
    # Flatten tensors
    pred_flat = predicted.view(-1)
    target_flat = target.view(-1)
    
    # Compute intersection and sums
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    
    # Dice coefficient
    dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Return loss (1 - dice)
    return 1 - dice


def dice_score(predicted, target, smooth=1e-6):
    """
    Dice similarity coefficient (for evaluation, not loss).
    
    Args:
        predicted: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        smooth: Smoothing factor
    
    Returns:
        Dice score in [0, 1], where higher is better
    """
    return 1 - dice_loss(predicted, target, smooth)


def batch_dice_loss(predictions, targets):
    """
    Average Dice loss across a batch.
    
    Args:
        predictions: List of predicted masks
        targets: List of target masks
    
    Returns:
        Average Dice loss
    """
    losses = []
    for pred, target in zip(predictions, targets):
        loss = dice_loss(pred, target)
        losses.append(loss)
    
    return torch.stack(losses).mean()


if __name__ == "__main__":
    # Test loss function
    import numpy as np
    
    # Perfect match
    pred = torch.tensor(np.random.rand(100, 100).astype(np.float32))
    target = (pred > 0.5).float()
    
    loss = dice_loss(pred, target)
    score = dice_score(pred, target)
    
    print("Perfect match test:")
    print(f"  Dice loss: {loss:.4f} (should be close to 0)")
    print(f"  Dice score: {score:.4f} (should be close to 1)")
    
    # Random prediction
    pred_random = torch.rand(100, 100)
    target_random = (torch.rand(100, 100) > 0.5).float()
    
    loss_random = dice_loss(pred_random, target_random)
    score_random = dice_score(pred_random, target_random)
    
    print("\nRandom prediction test:")
    print(f"  Dice loss: {loss_random:.4f}")
    print(f"  Dice score: {score_random:.4f}")
