# FILE: visualize_training.py

"""
Visualization utilities for SegmentationViNT training
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import os
from typing import Optional, List
from PIL import Image

# --- Placeholder for imports from your existing utils ---
# These are assumed to be defined in your project's visualize_utils
VIZ_IMAGE_SIZE = (224, 224) # Default size for visualization consistency
RED = 'red'
GREEN = 'green'
BLUE = 'blue'
CYAN = 'cyan'
YELLOW = 'yellow'
MAGENTA = 'magenta'

def numpy_to_img(arr: np.ndarray) -> Image.Image:
    """Converts a numpy array (C, H, W) to a PIL Image."""
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    if arr.shape[0] == 3: # Handle CHW format
        arr = arr.transpose(1, 2, 0)
    return Image.fromarray(arr)

def plot_trajs_and_points(ax, trajs, points, traj_colors, point_colors, traj_labels, point_labels):
    """Plots trajectories and points on a matplotlib axis."""
    for traj, color, label in zip(trajs, traj_colors, traj_labels):
        ax.plot(traj[:, 0], traj[:, 1], color=color, label=label, marker='.')
    for point, color, label in zip(points, point_colors, point_labels):
        ax.scatter(point[0], point[1], color=color, label=label, s=100, marker='*' if 'Goal' in label else 'o', zorder=10)
    ax.legend()
# --- End of placeholders ---


# Define the color palette for your navigation classes
SEG_COLORS = {
    0: [0, 255, 0],     # floor - green
    1: [255, 0, 0],     # wall - red  
    2: [255, 255, 0],   # door - yellow
    3: [0, 0, 255],     # furniture - blue
    4: [128, 128, 128], # unknown - gray
}

def visualize_segmentation_predictions(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    batch_seg_preds: np.ndarray,
    batch_seg_labels: Optional[np.ndarray],
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    batch_goals: np.ndarray,
    save_folder: str,
    epoch: int,
    eval_type: str = "val",
    num_images: int = 8,
    use_wandb: bool = True,
    class_names: List[str] = ['floor', 'wall', 'door', 'furniture', 'unknown'],
):
    """
    Main function to generate and log a grid of visualizations for a batch.
    """
    visualize_path = os.path.join(save_folder, "visualize", eval_type, f"epoch_{epoch}")
    os.makedirs(visualize_path, exist_ok=True)
    
    batch_size = min(batch_obs_images.shape[0], num_images)
    wandb_images = []
    
    for i in range(batch_size):
        # Prepare data for a single sample
        obs_img = numpy_to_img(batch_obs_images[i, -3:, :, :]) # Get last frame from context
        goal_img = numpy_to_img(batch_goal_images[i])
        
        # Convert segmentation predictions/labels to colored images
        seg_pred_colored = segmentation_to_color(batch_seg_preds[i], SEG_COLORS)
        
        seg_label_colored = None
        if batch_seg_labels is not None:
            seg_label_colored = segmentation_to_color(batch_seg_labels[i], SEG_COLORS)
        
        # Create the comprehensive visualization plot
        save_path = os.path.join(visualize_path, f"sample_{i:04d}.png")
        create_segmentation_visualization(
            obs_img=obs_img,
            goal_img=goal_img,
            seg_pred=seg_pred_colored,
            seg_label=seg_label_colored,
            pred_waypoints=batch_pred_waypoints[i],
            label_waypoints=batch_label_waypoints[i],
            goal_pos=batch_goals[i],
            class_names=class_names,
            save_path=save_path,
        )
        
        if use_wandb:
            wandb_images.append(wandb.Image(save_path, caption=f"Epoch {epoch}, Sample {i}"))
    
    if use_wandb and wandb_images:
        wandb.log({f"{eval_type}_visualizations": wandb_images}, commit=False)

def segmentation_to_color(seg_mask: np.ndarray, color_map: dict) -> np.ndarray:
    """Convert a segmentation mask (class indices) to a colored RGB image."""
    # If logits are passed (C, H, W), convert to class indices first
    if seg_mask.ndim == 3:
        seg_mask = np.argmax(seg_mask, axis=0)
    
    h, w = seg_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in color_map.items():
        colored[seg_mask == class_idx] = color
    
    return colored

def create_segmentation_visualization(
    obs_img: Image.Image,
    goal_img: Image.Image,
    seg_pred: np.ndarray,
    seg_label: Optional[np.ndarray],
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    goal_pos: np.ndarray,
    class_names: List[str],
    save_path: str,
):
    """Create a multi-panel plot visualizing all aspects of a single prediction."""
    
    num_cols = 5 if seg_label is not None else 4
    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 4, 8))
    
    # --- Top Row: Images and Legend ---
    axes[0, 0].imshow(obs_img); axes[0, 0].set_title("Observation RGB"); axes[0, 0].axis('off')
    axes[0, 1].imshow(seg_pred); axes[0, 1].set_title("Predicted Seg"); axes[0, 1].axis('off')
    
    col_offset = 0
    if seg_label is not None:
        axes[0, 2].imshow(seg_label); axes[0, 2].set_title("Ground Truth Seg"); axes[0, 2].axis('off')
        col_offset = 1
        
    axes[0, 2+col_offset].imshow(goal_img); axes[0, 2+col_offset].set_title("Goal RGB"); axes[0, 2+col_offset].axis('off')
    
    # --- Bottom Row: Trajectories and Stats ---
    axes[1, 0].imshow(obs_img); plot_trajectory_on_image(axes[1, 0], pred_waypoints, label_waypoints); axes[1, 0].set_title("Traj on RGB"); axes[1, 0].axis('off')
    axes[1, 1].imshow(seg_pred); plot_trajectory_on_image(axes[1, 1], pred_waypoints, label_waypoints); axes[1, 1].set_title("Traj on Seg"); axes[1, 1].axis('off')
    
    # Trajectory Plot
    plot_trajs_and_points(
        axes[1, 2], [pred_waypoints, label_waypoints], [np.array([0, 0]), goal_pos],
        [CYAN, MAGENTA], [GREEN, RED], ["Predicted", "Ground Truth"], ["Start", "Goal"]
    )
    axes[1, 2].set_title("Trajectory Plot"); axes[1, 2].grid(True, alpha=0.3); axes[1, 2].axis('equal')

    # Statistics Text
    stats_ax = axes[1, 3]
    stats_ax.axis('off'); stats_ax.set_title("Statistics")
    traj_error = np.mean(np.linalg.norm(pred_waypoints - label_waypoints, axis=1))
    stats_text = f"Trajectory Error: {traj_error:.3f}"
    stats_ax.text(0.05, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def plot_trajectory_on_image(ax, pred_waypoints, label_waypoints):
    """Helper to plot trajectories on top of an image."""
    h, w = VIZ_IMAGE_SIZE
    # This is a simplified projection. You may need to adjust scale and center.
    scale = 40 
    center_x, center_y = w // 2, h - 10

    def to_pixels(waypoints):
        pixels = np.zeros_like(waypoints)
        pixels[:, 0] = center_x + waypoints[:, 0] * scale
        pixels[:, 1] = center_y - waypoints[:, 1] * scale # Y is inverted in image coordinates
        return pixels

    pred_px = to_pixels(pred_waypoints)
    label_px = to_pixels(label_waypoints)
    
    ax.plot(pred_px[:, 0], pred_px[:, 1], color=CYAN, linewidth=3, alpha=0.8, label='Predicted')
    ax.plot(label_px[:, 0], label_px[:, 1], color=MAGENTA, linewidth=2, linestyle='--', alpha=0.8, label='Ground Truth')
