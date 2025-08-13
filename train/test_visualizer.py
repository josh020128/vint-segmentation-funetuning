# FILE: test_visualizer.py

"""
A standalone script to test the visualization utilities in visualize_segmentation.py
by generating and plotting fake data.
"""
import torch
import numpy as np
import os

# Make sure this import path matches your project structure
from vint_train.visualizing.visualize_segmentation import visualize_segmentation_predictions

def create_dummy_data(batch_size: int = 4, h: int = 128, w: int = 160, context: int = 5, num_waypoints: int = 8, num_classes: int = 5):
    """Creates a batch of fake data with realistic shapes and value ranges."""
    print("Generating dummy data for visualization test...")
    
    # 1. Image Data (C, H, W)
    # obs_images are a stack of context frames
    batch_obs_images = torch.rand(batch_size, 3 * context, h, w)
    batch_goal_images = torch.rand(batch_size, 3, h, w)
    
    # 2. Segmentation Data
    # Predictions are logits (raw model output), labels are class indices
    batch_seg_preds = torch.randn(batch_size, num_classes, h, w)
    batch_seg_labels = torch.randint(0, num_classes, (batch_size, h, w))
    
    # 3. Navigation Data
    # Waypoints are in a normalized coordinate system, typically [-1, 1]
    batch_pred_waypoints = (torch.rand(batch_size, num_waypoints, 2) - 0.5) * 2
    batch_label_waypoints = (torch.rand(batch_size, num_waypoints, 2) - 0.5) * 2
    batch_goals = (torch.rand(batch_size, 2) - 0.5) * 4 # Goal can be further away
    
    # 4. Metadata
    dataset_indices = torch.arange(batch_size)
    
    print("✓ Dummy data generated successfully.")
    return {
        "obs_images": batch_obs_images,
        "goal_images": batch_goal_images,
        "seg_preds": batch_seg_preds,
        "seg_labels": batch_seg_labels,
        "pred_waypoints": batch_pred_waypoints,
        "label_waypoints": batch_label_waypoints,
        "goals": batch_goals,
        "dataset_indices": dataset_indices,
    }

def main():
    """Main function to run the visualization test."""
    print("\n" + "="*60)
    print("STARTING VISUALIZATION TEST")
    print("="*60)
    
    # Configuration
    save_folder = "test_vis_outputs"
    epoch = 999 # Use a high number to indicate it's a test
    eval_type = "test_run"
    num_images_to_viz = 4
    
    # Create dummy data
    dummy_batch = create_dummy_data(batch_size=num_images_to_viz)
    
    # Convert tensors to numpy for the visualization function
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # Call the main visualization function from your script
    try:
        print("\nCalling visualize_segmentation_predictions...")
        visualize_segmentation_predictions(
            batch_obs_images=to_numpy(dummy_batch["obs_images"]),
            batch_goal_images=to_numpy(dummy_batch["goal_images"]),
            batch_seg_preds=to_numpy(dummy_batch["seg_preds"]),
            batch_seg_labels=to_numpy(dummy_batch["seg_labels"]),
            batch_pred_waypoints=to_numpy(dummy_batch["pred_waypoints"]),
            batch_label_waypoints=to_numpy(dummy_batch["label_waypoints"]),
            batch_goals=to_numpy(dummy_batch["goals"]),
            save_folder=save_folder,
            epoch=epoch,
            eval_type=eval_type,
            num_images=num_images_to_viz,
            use_wandb=False, # Don't log to wandb for this test
        )
        print("\n" + "="*60)
        print("✅ VISUALIZATION TEST SUCCEEDED!")
        print(f"Check the '{save_folder}' directory for the output images.")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ VISUALIZATION TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*60)


if __name__ == "__main__":
    main()
