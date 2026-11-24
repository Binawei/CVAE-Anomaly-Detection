"""
Soft thresholding function for differentiable optimization.
"""

import torch


def soft_threshold(x, threshold, steepness=10):
    """
    Differentiable approximation of hard thresholding.
    
    Hard threshold (non-differentiable):
        output = 1 if x > threshold else 0
    
    Soft threshold (differentiable):
        output = sigmoid(steepness × (x - threshold))
    
    Args:
        x: Input tensor (error values)
        threshold: Threshold value (learnable parameter)
        steepness: Controls smoothness (higher = closer to hard threshold)
    
    Returns:
        Tensor with values in [0, 1] (soft binary mask)
    """
    return torch.sigmoid(steepness * (x - threshold))


def hard_threshold(x, threshold):
    """
    Hard thresholding (for final evaluation, not training).
    
    Args:
        x: Input tensor
        threshold: Threshold value
    
    Returns:
        Binary tensor (0 or 1)
    """
    return (x > threshold).float()


if __name__ == "__main__":
    # Test soft vs hard thresholding
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create test data
    x = torch.linspace(0, 0.5, 100)
    threshold = torch.tensor(0.25)
    
    # Compute thresholds
    soft = soft_threshold(x, threshold, steepness=10)
    hard = hard_threshold(x, threshold)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x.numpy(), soft.numpy(), label='Soft Threshold (steepness=10)', linewidth=2)
    plt.plot(x.numpy(), hard.numpy(), label='Hard Threshold', linewidth=2, linestyle='--')
    plt.axvline(threshold.item(), color='red', linestyle=':', label=f'Threshold={threshold.item()}')
    plt.xlabel('Input Value (Error)')
    plt.ylabel('Output (Mask Value)')
    plt.title('Soft vs Hard Thresholding')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('soft_threshold_demo.png', dpi=150)
    print("Saved: soft_threshold_demo.png")
