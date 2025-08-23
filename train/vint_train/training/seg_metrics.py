# FILE: vint_train/training/seg_metrics.py

"""
Evaluation metrics for the dual-input "co-pilot" navigation model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from tqdm import tqdm

class NavigationMetrics:
    """
    Calculates key navigation metrics, including consistency with the primary
    ViNT planner, using ground truth segmentation masks for collision checking.
    """
    
    def __init__(self, 
                 success_threshold: float = 0.3, 
                 collision_margin: int = 5):
        """
        Initializes the NavigationMetrics tracker.

        Args:
            success_threshold (float): Max distance from goal to be a success.
            collision_margin (int): Pixel radius to check for obstacles around a point.
        """
        self.success_threshold = success_threshold
        self.collision_margin = collision_margin
        self.reset()

    def reset(self):
        """Resets all metric lists for a new evaluation run."""
        self.successes = []
        self.collisions = []
        self.goal_distances = []
        self.spl_scores = []
        # <<< ADDED: A list to track consistency with the ViNT planner >>>
        self.consistency_scores = []

    def update(self, outputs: Dict[str, torch.Tensor], true_waypoints: torch.Tensor, true_seg_masks: torch.Tensor):
        """
        Updates the metrics with data from a new batch.

        Args:
            outputs (Dict[str, torch.Tensor]): Predictions from the model, including
                                               'action_pred' and 'vint_trajectory'.
            true_waypoints (torch.Tensor): The ground truth trajectories.
            true_seg_masks (torch.Tensor): The ground truth segmentation masks.
        """
        pred_traj_batch = outputs['action_pred'].detach().cpu().numpy()
        vint_traj_batch = outputs['vint_trajectory'].detach().cpu().numpy()
        true_traj_batch = true_waypoints.detach().cpu().numpy()
        
        if true_seg_masks.dim() == 4:
             true_seg_masks = torch.argmax(true_seg_masks, dim=1)
        true_seg_batch = true_seg_masks.cpu().numpy()

        for i in range(pred_traj_batch.shape[0]):
            pred_traj = pred_traj_batch[i]
            true_traj = true_traj_batch[i]
            vint_traj = vint_traj_batch[i]
            true_seg = true_seg_batch[i]
            
            # --- Goal distance and Success ---
            goal_dist = np.linalg.norm(pred_traj[-1] - true_traj[-1])
            is_success = goal_dist < self.success_threshold
            self.goal_distances.append(goal_dist)
            self.successes.append(is_success)
            
            # --- Collision Check ---
            has_collision = self._check_collision(pred_traj, true_seg)
            self.collisions.append(has_collision)
            
            # --- Success weighted by Path Length (SPL) ---
            spl = 0.0
            if is_success and not has_collision:
                true_len = self._path_length(true_traj)
                pred_len = self._path_length(pred_traj)
                spl = true_len / max(true_len, pred_len, 1e-8)
            self.spl_scores.append(spl)

            # <<< ADDED: Calculate consistency metric >>>
            # Measures the average deviation from the original ViNT plan.
            deviation = np.mean(np.linalg.norm(pred_traj - vint_traj, axis=-1))
            # Normalize by max possible distance in a [-1, 1] space for a score from 0 to 1
            consistency = max(0.0, 1.0 - deviation / np.sqrt(8))
            self.consistency_scores.append(consistency)

    def _check_collision(self, trajectory: np.ndarray, seg_map: np.ndarray) -> bool:
        """Checks if a trajectory collides with any non-navigable space (non-floor)."""
        h, w = seg_map.shape
        traj_pixels_x = (trajectory[:, 1] + 1) / 2.0 * (w - 1)
        traj_pixels_y = (trajectory[:, 0] + 1) / 2.0 * (h - 1)
        
        for px, py in zip(traj_pixels_x, traj_pixels_y):
            y_min, y_max = max(0, int(py - self.collision_margin)), min(h, int(py + self.collision_margin + 1))
            x_min, x_max = max(0, int(px - self.collision_margin)), min(w, int(px + self.collision_margin + 1))
            region = seg_map[y_min:y_max, x_min:x_max]
            if region.size > 0 and np.any(region != 0):
                return True
        return False

    def _path_length(self, trajectory: np.ndarray) -> float:
        """Calculates the total length of a trajectory."""
        if len(trajectory) < 2: return 0.0
        return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

    def get_metrics(self) -> Dict[str, float]:
        """Computes and returns the final summary of all metrics."""
        safe_success = [s and not c for s, c in zip(self.successes, self.collisions)]
        metrics = {
            'nav/success_rate': np.mean(self.successes) if self.successes else 0.0,
            'nav/collision_rate': np.mean(self.collisions) if self.collisions else 0.0,
            'nav/mean_goal_distance': np.mean(self.goal_distances) if self.goal_distances else 0.0,
            'nav/spl': np.mean(self.spl_scores) if self.spl_scores else 0.0,
            'nav/safe_success_rate': np.mean(safe_success) if safe_success else 0.0,
            # <<< ADDED: Report the new consistency score >>>
            'nav/vint_consistency': np.mean(self.consistency_scores) if self.consistency_scores else 0.0,
        }
        return metrics

def evaluate_segmentation_vint(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict,
) -> Dict[str, float]:
    """Main evaluation function for the co-pilot SegmentationViNT."""
    model.eval()
    
    metrics = NavigationMetrics(
        success_threshold=config.get("success_threshold", 0.3),
        collision_margin=config.get("collision_margin", 5)
    )
    
    num_seg_classes = model.module.num_seg_classes if hasattr(model, 'module') else model.num_seg_classes
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        obs_images = batch['obs_images'].to(device)
        goal_images = batch['goal_images'].to(device)
        true_waypoints = batch['actions'].to(device)
        obs_seg_mask_one_hot = batch['obs_seg_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(obs_images, goal_images, obs_seg_mask_one_hot)
        
        true_seg_labels = torch.argmax(obs_seg_mask_one_hot, dim=1)

        # <<< MODIFIED: Pass the full model output to the update function >>>
        metrics.update(
            outputs=outputs,
            true_waypoints=true_waypoints,
            true_seg_masks=true_seg_labels,
        )
    
    model.train()
    return metrics.get_metrics()
