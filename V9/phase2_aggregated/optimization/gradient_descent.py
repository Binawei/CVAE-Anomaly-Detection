"""
Gradient Descent optimization for learning thresholds and weights.
Main optimization method as specified by supervisor.
"""

import torch
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from combine_maps import combine_error_maps_soft
from losses import dice_loss
from metrics import compute_all_metrics


class GradientDescentOptimizer:
    """
    Optimize threshold and weighting parameters using gradient descent.
    """
    
    def __init__(self, train_data, val_data):
        """
        Args:
            train_data: List of training case dicts
            val_data: List of validation case dicts
        """
        self.train_data = train_data
        self.val_data = val_data
        
        # Initialize parameters
        self.params = {
            'threshold_img': torch.tensor(config.INITIAL_PARAMS['threshold_img'], 
                                         requires_grad=True, device=config.DEVICE),
            'threshold_seg': torch.tensor(config.INITIAL_PARAMS['threshold_seg'],
                                         requires_grad=True, device=config.DEVICE),
            'alpha': torch.tensor(config.INITIAL_PARAMS['alpha'],
                                 requires_grad=True, device=config.DEVICE)
        }
        
        # Optimizer (Adam with learning rate scheduling)
        self.optimizer = torch.optim.Adam([
            self.params['threshold_img'],
            self.params['threshold_seg'],
            self.params['alpha']
        ], lr=config.LEARNING_RATE)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'threshold_img': [],
            'threshold_seg': [],
            'alpha': []
        }
        
        self.best_val_loss = float('inf')
        self.best_params = None
        self.patience_counter = 0
    
    def clamp_parameters(self):
        """Clamp parameters to valid ranges."""
        with torch.no_grad():
            self.params['threshold_img'].clamp_(*config.PARAM_BOUNDS['threshold_img'])
            self.params['threshold_seg'].clamp_(*config.PARAM_BOUNDS['threshold_seg'])
            self.params['alpha'].clamp_(*config.PARAM_BOUNDS['alpha'])
    
    def compute_loss(self, data_list):
        """Compute average Dice loss across cases."""
        total_loss = 0
        
        for data in data_list:
            # Combine error maps with current parameters
            combined = combine_error_maps_soft(
                data['img_error'],
                data['seg_uncertainty'],
                self.params['threshold_img'],
                self.params['threshold_seg'],
                self.params['alpha'],
                steepness=config.STEEPNESS
            )
            
            # Compute loss
            loss = dice_loss(combined, data['annotation'])
            total_loss += loss
        
        return total_loss / len(data_list)
    
    def train_epoch(self):
        """Run one training epoch."""
        self.optimizer.zero_grad()
        
        # Compute loss on training data
        train_loss = self.compute_loss(self.train_data)
        
        # Backpropagation
        train_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Clamp to valid ranges
        self.clamp_parameters()
        
        return train_loss.item()
    
    def validate(self):
        """Evaluate on validation data."""
        with torch.no_grad():
            val_loss = self.compute_loss(self.val_data)
        return val_loss.item()
    
    def train(self):
        """
        Main training loop with early stopping.
        
        Returns:
            Best parameters found
        """
        print("\n" + "="*80)
        print("GRADIENT DESCENT OPTIMIZATION")
        print("="*80)
        print(f"Initial parameters:")
        print(f"  threshold_img: {self.params['threshold_img'].item():.4f}")
        print(f"  threshold_seg: {self.params['threshold_seg'].item():.4f}")
        print(f"  alpha: {self.params['alpha'].item():.4f}")
        print(f"\nTraining for up to {config.MAX_EPOCHS} epochs...")
        print(f"Early stopping patience: {config.PATIENCE}")
        print("="*80)
        
        for epoch in range(config.MAX_EPOCHS):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['threshold_img'].append(self.params['threshold_img'].item())
            self.history['threshold_seg'].append(self.params['threshold_seg'].item())
            self.history['alpha'].append(self.params['alpha'].item())
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{config.MAX_EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"θ_img: {self.params['threshold_img'].item():.3f} | "
                      f"θ_seg: {self.params['threshold_seg'].item():.3f} | "
                      f"α: {self.params['alpha'].item():.3f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss - config.MIN_DELTA:
                self.best_val_loss = val_loss
                self.best_params = {
                    'threshold_img': self.params['threshold_img'].item(),
                    'threshold_seg': self.params['threshold_seg'].item(),
                    'alpha': self.params['alpha'].item()
                }
                self.patience_counter = 0
                print(f"  → New best! Val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"No improvement for {config.PATIENCE} epochs")
                break
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best parameters:")
        print(f"  threshold_img: {self.best_params['threshold_img']:.4f}")
        print(f"  threshold_seg: {self.best_params['threshold_seg']:.4f}")
        print(f"  alpha: {self.best_params['alpha']:.4f}")
        print("="*80)
        
        return self.best_params, self.history


def run_gradient_descent(train_data, val_data):
    """
    Convenience function to run gradient descent optimization.
    
    Args:
        train_data: Training cases
        val_data: Validation cases
    
    Returns:
        Tuple of (best_params, history)
    """
    optimizer = GradientDescentOptimizer(train_data, val_data)
    best_params, history = optimizer.train()
    
    return best_params, history


if __name__ == "__main__":
    # Test with dummy data
    print("Testing gradient descent optimizer...")
    
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
    
    best_params, history = run_gradient_descent(dummy_train, dummy_val)
    print("\nTest completed successfully!")
