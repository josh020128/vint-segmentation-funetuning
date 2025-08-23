# FILE: vint_train/training/seg_losses.py

"""
Loss function for the dual-input ViNT model with a "safety co-pilot" architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class CoPilotNavigationLoss(nn.Module):
    """
    Calculates the loss for the co-pilot model, balancing navigation accuracy
    with consistency to the primary ViNT planner.
    """
    def __init__(
        self,
        action_loss_weight: float = 1.0,
        dist_loss_weight: float = 0.5,
        consistency_weight: float = 0.2, # Weight for the new consistency term
    ):
        """
        Initializes the loss module.

        Args:
            action_loss_weight (float): Weight for the final trajectory's MSE loss.
            dist_loss_weight (float): Weight for the distance-to-goal MSE loss.
            consistency_weight (float): Weight to encourage the final trajectory
                                        to stay close to the ViNT's original plan.
        """
        super().__init__()
        self.action_loss_weight = action_loss_weight
        self.dist_loss_weight = dist_loss_weight
        self.consistency_weight = consistency_weight
        
        # Use standard Mean Squared Error for the losses
        self.action_loss_fn = nn.MSELoss()
        self.dist_loss_fn = nn.MSELoss()
        self.consistency_loss_fn = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the combined navigation and consistency loss.

        Args:
            outputs (Dict[str, torch.Tensor]): Predictions from the model, must include
                                               'action_pred', 'dist_pred', and 'vint_trajectory'.
            targets (Dict[str, torch.Tensor]): Ground truth labels, must include
                                               'actions' and 'distance'.
            action_mask (Optional[torch.Tensor]): A mask to exclude invalid actions.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of all computed loss components.
        """
        
        # --- 1. Action Loss ---
        # Compares the FINAL trajectory to the ground truth
        action_loss = self.action_loss_fn(outputs['action_pred'], targets['actions'])
        
        # --- 2. Distance Loss ---
        dist_loss = self.dist_loss_fn(outputs['dist_pred'].squeeze(), targets['distance'].float())

        # --- 3. Consistency Loss (CRITICAL for Co-Pilot Model) ---
        # Compares the final trajectory to the ViNT's ORIGINAL plan.
        # This prevents the safety module from overpowering the main planner.
        consistency_loss = self.consistency_loss_fn(outputs['action_pred'], outputs['vint_trajectory'])
        
        # --- 4. Combine All Losses ---
        total_loss = (self.action_loss_weight * action_loss) + \
                     (self.dist_loss_weight * dist_loss) + \
                     (self.consistency_weight * consistency_loss)
        
        # Apply action mask if provided (useful for filtering out certain samples)
        if action_mask is not None:
            total_loss = (total_loss * action_mask).sum() / (action_mask.sum() + 1e-8)
        
        return {
            'total_loss': total_loss,
            'action_loss': action_loss.detach(),
            'dist_loss': dist_loss.detach(),
            'consistency_loss': consistency_loss.detach(),
        }
