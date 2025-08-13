# FILE: vint_train/training/seg_losses.py

"""
Loss functions for segmentation-enhanced navigation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class FocalLoss(nn.Module):
    """Focal loss that correctly handles an ignore_index."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        mask = (targets != self.ignore_index)
        return focal_loss[mask].mean()

class SegmentationNavigationLoss(nn.Module):
    """
    Combines navigation and segmentation losses with optional uncertainty weighting.
    """
    def __init__(
        self,
        action_loss_weight: float = 1.0,
        dist_loss_weight: float = 0.5,
        seg_loss_weight: float = 0.3,
        use_focal_loss: bool = True,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        self.action_loss_weight = action_loss_weight
        self.dist_loss_weight = dist_loss_weight
        self.seg_loss_weight = seg_loss_weight
        
        self.action_loss_fn = nn.MSELoss(reduction='none')
        self.dist_loss_fn = nn.MSELoss()
        
        if use_focal_loss:
            self.seg_loss_fn = FocalLoss()
        else:
            self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        
        if use_uncertainty_weighting:
            self.log_action_var = nn.Parameter(torch.zeros(1))
            self.log_dist_var = nn.Parameter(torch.zeros(1))
            self.log_seg_var = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred_actions: torch.Tensor,
        true_actions: torch.Tensor,
        pred_dist: torch.Tensor,
        true_dist: torch.Tensor,
        pred_seg: torch.Tensor,
        true_seg: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # Action loss (masked for invalid samples)
        action_loss_per_element = self.action_loss_fn(pred_actions, true_actions).mean(dim=-1)
        if action_mask is not None:
            action_mask = action_mask.unsqueeze(1)
            action_loss = (action_loss_per_element * action_mask).sum() / (action_mask.sum() + 1e-8)
        else:
            action_loss = action_loss_per_element.mean()
        
        # Distance loss
        dist_loss = self.dist_loss_fn(pred_dist, true_dist)
        
        # Segmentation loss
        seg_loss = torch.tensor(0.0, device=pred_actions.device)
        if true_seg is not None and pred_seg is not None:
            if torch.any(true_seg != 255): 
                seg_loss = self.seg_loss_fn(pred_seg, true_seg)
        
        # Combine losses
        if self.use_uncertainty_weighting:
            # <<< FIXED: Ensure learnable parameters are on the same device as the inputs >>>
            device = pred_actions.device
            log_action_var = self.log_action_var.to(device)
            log_dist_var = self.log_dist_var.to(device)
            log_seg_var = self.log_seg_var.to(device)

            action_precision = torch.exp(-log_action_var)
            dist_precision = torch.exp(-log_dist_var)
            seg_precision = torch.exp(-log_seg_var)
            
            total_loss = (action_precision * action_loss + log_action_var) + \
                         (dist_precision * dist_loss + log_dist_var) + \
                         (seg_precision * seg_loss + log_seg_var)
            total_loss /= 3.0
        else:
            total_loss = (self.action_loss_weight * action_loss) + \
                         (self.dist_loss_weight * dist_loss) + \
                         (self.seg_loss_weight * seg_loss)
        
        return {
            'total_loss': total_loss,
            'action_loss': action_loss.detach(),
            'dist_loss': dist_loss.detach(),
            'seg_loss': seg_loss.detach(),
        }
