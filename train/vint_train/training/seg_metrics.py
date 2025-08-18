# FILE: vint_train/training/seg_metrics.py

"""
Evaluation metrics for segmentation-enhanced navigation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage  # Use scipy for fast distance transform
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

class SegmentationMetrics:
    """Calculates standard segmentation metrics like mIoU."""
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        mask = target != self.ignore_index
        pred_flat = pred[mask]
        target_flat = target[mask]
        
        indices = self.num_classes * target_flat + pred_flat
        cm = np.bincount(indices.numpy(), minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += cm

    def get_metrics(self) -> Dict[str, float]:
        """ <<< REVISED: Now calculates and returns IoU for each class. >>> """
        metrics = {}
        
        # Use np.diag to get all true positives at once
        true_positives = np.diag(self.confusion_matrix)
        # Use .sum() to get false positives and false negatives
        false_positives = self.confusion_matrix.sum(axis=0) - true_positives
        false_negatives = self.confusion_matrix.sum(axis=1) - true_positives
        
        # Calculate IoU for each class, avoiding division by zero
        denominator = true_positives + false_positives + false_negatives
        iou = np.divide(true_positives, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator!=0)
        
        for i in range(self.num_classes):
            metrics[f'iou_class_{i}'] = iou[i]
        
        # Mean IoU
        metrics['mIoU'] = np.nanmean(iou)
        
        # Pixel accuracy
        total_correct = true_positives.sum()
        total_pixels = self.confusion_matrix.sum()
        metrics['pixel_accuracy'] = total_correct / (total_pixels + 1e-8)
        
        return metrics

class NavigationMetrics:
    """Calculates standard navigation metrics like success rate."""
    def __init__(self, success_threshold: float = 0.5):
        self.success_threshold = success_threshold
        self.reset()

    def reset(self):
        self.success = []
        self.collisions = []
        self.goal_distances = []
        self.path_lengths_ratio = []
        self.spl_scores = []

    def update(self, pred_waypoints: torch.Tensor, true_waypoints: torch.Tensor, seg_pred: torch.Tensor, obstacle_classes: List[int]):
        pred_traj = pred_waypoints.detach().cpu().numpy()
        true_traj = true_waypoints.detach().cpu().numpy()
        seg_pred = seg_pred.detach().cpu()

        for i in range(pred_traj.shape[0]):
            goal_dist = np.linalg.norm(pred_traj[i, -1] - true_traj[i, -1])
            self.goal_distances.append(goal_dist)
            self.success.append(goal_dist < self.success_threshold)
            
            # Check for collisions
            has_collision = self._check_collision(pred_traj[i], seg_pred[i], obstacle_classes)
            self.collisions.append(has_collision)

            is_success = goal_dist < self.success_threshold
            if is_success:
                true_len = np.sum(np.linalg.norm(true_traj[1:] - true_traj[:-1], axis=-1))
                pred_len = np.sum(np.linalg.norm(pred_traj[1:] - pred_traj[:-1], axis=-1))
                spl = true_len / max(true_len, pred_len, 1e-8)
                self.spl_scores.append(spl)
            else:
                self.spl_scores.append(0.0)

    def _check_collision(self, trajectory: np.ndarray, seg_map: torch.Tensor, obstacle_classes: List[int]) -> bool:
        if seg_map.dim() == 3:
            seg_map = torch.argmax(seg_map, dim=0)
        
        h, w = seg_map.shape
        pixel_coords = np.zeros_like(trajectory, dtype=int)
        pixel_coords[:, 0] = np.clip((trajectory[:, 0] + 1) * w / 2, 0, w - 1)
        # The y-coordinate now starts from the middle of the image (h/2) and goes down.
        pixel_coords[:, 1] = np.clip(h / 2 + (trajectory[:, 1] * h / 2), 0, h - 1)
        
        for x, y in pixel_coords:
            if seg_map[y, x] != 0: #floor index
                return True # Collision if not on the floor
        return False

    def get_metrics(self) -> Dict[str, float]:
        return {
            'success_rate': np.mean(self.success) if self.success else 0.0,
            'collision_rate': np.mean(self.collisions) if self.collisions else 0.0,
            'mean_goal_distance': np.mean(self.goal_distances) if self.goal_distances else 0.0,
            'spl': np.mean(self.spl_scores) if self.spl_scores else 0.0,
        }

class CombinedMetrics:
    """Combines all metrics for a comprehensive evaluation."""
    def __init__(self, num_seg_classes: int, **kwargs):
        self.obstacle_classes = kwargs.get('obstacle_classes', [1, 3, 4]) # wall, furniture, and unknown
        self.walkable_classes = kwargs.get('walkable_classes', [0])   # floor
        self.seg_metrics = SegmentationMetrics(num_seg_classes)
        self.nav_metrics = NavigationMetrics(**kwargs)
        self.reset()

    def reset(self):
        self.seg_metrics.reset()
        self.nav_metrics.reset()
        self.semantic_consistency_scores = []
        self.obstacle_awareness_scores = []

    def update(self, pred_waypoints: torch.Tensor, true_waypoints: torch.Tensor, pred_seg: torch.Tensor, true_seg: Optional[torch.Tensor] = None):
        self.nav_metrics.update(pred_waypoints, true_waypoints, pred_seg, self.obstacle_classes)
        if true_seg is not None:
            self.seg_metrics.update(pred_seg, true_seg)
        
        self.semantic_consistency_scores.append(self._calculate_semantic_consistency(pred_waypoints, pred_seg))
        self.obstacle_awareness_scores.append(self._calculate_obstacle_awareness(pred_waypoints, pred_seg))

    def _calculate_semantic_consistency(self, waypoints: torch.Tensor, seg_pred: torch.Tensor) -> float:
        """ <<< REVISED: Fast, vectorized consistency calculation. >>> """
        seg_probs = F.softmax(seg_pred, dim=1)
        walkability_map = torch.sum(seg_probs[:, self.walkable_classes], dim=1, keepdim=True)
        
        grid = waypoints.unsqueeze(2)
        walkability_scores = F.grid_sample(walkability_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return walkability_scores.mean().item()

    def _calculate_obstacle_awareness(self, waypoints: torch.Tensor, seg_pred: torch.Tensor) -> float:
        """ <<< REVISED: Fast, vectorized obstacle awareness calculation. >>> """
        batch_scores = []
        for i in range(waypoints.shape[0]):
            seg = torch.argmax(seg_pred[i], dim=0).cpu()
            obstacle_mask = torch.zeros_like(seg, dtype=bool)
            for obs_class in self.obstacle_classes:
                obstacle_mask |= (seg == obs_class)
            
            if not obstacle_mask.any():
                batch_scores.append(1.0)
                continue
            
            dist_transform = scipy.ndimage.distance_transform_edt(~obstacle_mask.numpy())
            h, w = dist_transform.shape
            
            traj = waypoints[i].cpu().numpy()
            pixel_coords = np.zeros_like(traj)
            pixel_coords[:, 1] = np.clip((traj[:, 0] + 1) * w / 2, 0, w - 1)
            pixel_coords[:, 0] = np.clip((1 - traj[:, 1]) * h / 2, 0, h - 1)
            
            distances_on_path = scipy.ndimage.map_coordinates(dist_transform, pixel_coords.T, order=1)
            normalized_distances = np.clip(distances_on_path / 20.0, 0, 1.0)
            batch_scores.append(np.mean(normalized_distances) if normalized_distances.size > 0 else 0.0)
        
        return np.mean(batch_scores) if batch_scores else 0.0

    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        metrics.update({f'nav/{k}': v for k, v in self.nav_metrics.get_metrics().items()})
        metrics.update({f'seg/{k}': v for k, v in self.seg_metrics.get_metrics().items()})
        metrics['combined/semantic_consistency'] = np.mean(self.semantic_consistency_scores) if self.semantic_consistency_scores else 0.0
        metrics['combined/obstacle_awareness'] = np.mean(self.obstacle_awareness_scores) if self.obstacle_awareness_scores else 0.0
        return metrics

def evaluate_segmentation_vint(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_seg_classes: int,
) -> Dict[str, float]:
    """Main evaluation function for SegmentationViNT."""
    model.eval()
    metrics = CombinedMetrics(num_seg_classes=num_seg_classes)
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        obs_images = batch['obs_images'].to(device)
        goal_images = batch['goal_images'].to(device)
        true_waypoints = batch['actions'].to(device)
        true_seg = batch.get('obs_seg_mask', None)
        if true_seg is not None:
            true_seg = true_seg.to(device)
        
        with torch.no_grad():
            outputs = model(obs_images, goal_images)
        
        metrics.update(
            pred_waypoints=outputs['action_pred'],
            true_waypoints=true_waypoints,
            pred_seg=outputs['obs_seg_logits'],
            true_seg=true_seg,
        )
    
    return metrics.get_metrics()
