# FILE: vint_train/data/segmentation_dataset.py

"""
Dataset that loads navigation data with pre-computed semantic segmentation masks
for the dual-input "co-pilot" ViNT model.
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from typing import Optional, Dict, List
import os

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.data_utils import get_data_path, calculate_sin_cos

class ViNTSegmentationDataset(ViNT_Dataset):
    """
    Loads (RGB image stack, latest segmentation mask) pairs for the co-pilot model.
    It expects pre-computed segmentation masks to be available for each RGB frame.
    """
    
    def __init__(
        self,
        *args,
        is_train: bool = True, 
        seg_data_folder: str, # This is now required
        seg_model_name: str = "scand",
        seg_augmentation_prob: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.is_train = is_train    
        self.seg_data_folder = seg_data_folder
        self.seg_model_name = seg_model_name
        self.seg_augmentation_prob = seg_augmentation_prob
        
        if not self.seg_data_folder or not os.path.exists(self.seg_data_folder):
            raise FileNotFoundError(
                f"The 'seg_data_folder' is required and was not found at: {self.seg_data_folder}"
            )

        # Setup segmentation classes
        self.setup_segmentation_classes()
        
        # Log configuration
        print(f"\n{'='*60}")
        print("ViNTSegmentationDataset (Co-Pilot) Configuration:")
        print(f"  - Dataset: {kwargs.get('dataset_name', 'unknown')}")
        print(f"  - seg_data_folder: {seg_data_folder}")
        print(f"  - num_classes: {self.num_classes}")
        print(f"  - class_names: {self.class_names}")
        
        if len(self.index_to_data) > 0:
            self._test_segmentation_loading()
        print(f"{'='*60}\n")
    
    def train(self):
        """Set the dataset to training mode."""
        self.is_train = True

    def eval(self):
        """Set the dataset to evaluation mode."""
        self.is_train = False

    def setup_segmentation_classes(self):
        """Define segmentation classes based on the dataset name."""
        if self.seg_model_name == "scand":
            self.num_classes = 5
            self.class_names = ['floor', 'wall', 'door', 'furniture', 'unknown']
        else: # Fallback
            self.num_classes = 3
            self.class_names = ['navigable', 'obstacle', 'unknown']
    
    def get_seg_mask_path(self, image_path: str) -> str:
        """Convert an RGB image path to its corresponding segmentation mask path."""
        rel_path = os.path.relpath(image_path, self.data_folder)
        name_without_ext = os.path.splitext(os.path.basename(rel_path))[0]
        seg_filename = f"{name_without_ext}_seg.png"
        return os.path.join(self.seg_data_folder, os.path.dirname(rel_path), seg_filename)
    
    def load_segmentation_mask(self, image_path: str) -> torch.Tensor:
        """Loads a single segmentation mask from a file."""
        seg_path = self.get_seg_mask_path(image_path)
        try:
            if os.path.exists(seg_path):
                mask_pil = Image.open(seg_path).convert('L')
                resized_mask = mask_pil.resize(
                    (self.image_size[1], self.image_size[0]), 
                    Image.NEAREST
                )
                mask_np = np.array(resized_mask)
                return torch.from_numpy(mask_np.astype(np.int64))
            else:
                return torch.full(self.image_size, self.num_classes - 1, dtype=torch.long)
        except Exception as e:
            print(f"Error loading mask {seg_path}: {e}. Returning an 'unknown' mask.")
            return torch.full(self.image_size, self.num_classes - 1, dtype=torch.long)

    def _test_segmentation_loading(self):
        """Tests if segmentation loading works for a sample image."""
        print("\nTesting segmentation loading...")
        f_curr, curr_time, _ = self.index_to_data[0]
        test_path = get_data_path(self.data_folder, f_curr, curr_time)
        seg_path = self.get_seg_mask_path(test_path)
        print(f"  - Test image path: {test_path}")
        print(f"  - Expected seg path: {seg_path} (Exists: {os.path.exists(seg_path)})")
        seg_mask = self.load_segmentation_mask(test_path)
        print(f"  - Loaded seg shape: {seg_mask.shape}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves an item, including the RGB image stack and the latest segmentation mask.
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[idx]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)
        
        # <<< RUNTIME ERROR FIX: Corrected the range to load `context_size + 1` frames >>>
        # The model expects the current frame plus `context_size` past frames.
        context_times = list(range(
            curr_time - self.context_size * self.waypoint_spacing,
            curr_time + 1,
            self.waypoint_spacing
        ))
        
        # Load the current frame's segmentation mask
        current_img_path = get_data_path(self.data_folder, f_curr, curr_time)
        latest_seg_mask = self.load_segmentation_mask(current_img_path)
        
        # One-hot encode the latest mask for the model's seg_encoder
        obs_seg_mask_one_hot = F.one_hot(latest_seg_mask.long(), num_classes=self.num_classes)
        obs_seg_mask_one_hot = obs_seg_mask_one_hot.permute(2, 0, 1).float() # (C, H, W)

        # Load the stack of RGB images
        obs_images_list = [self._load_image_and_resize(get_data_path(self.data_folder, f_curr, t)) for t in context_times]
        obs_images = torch.cat(obs_images_list)

        # Load goal image
        goal_path = get_data_path(self.data_folder, f_goal, goal_time)
        goal_image = self._load_image_and_resize(goal_path)
        
        # Compute navigation data
        curr_traj_data = self._get_trajectory(f_curr)
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        distance = self.max_dist_cat if goal_is_negative else (goal_time - curr_time) // self.waypoint_spacing
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        
        action_mask = torch.tensor(
            (self.min_action_distance < distance < self.max_action_distance) and not goal_is_negative,
            dtype=torch.float32
        )
        
        return {
            'obs_images': obs_images.float(),
            'goal_images': goal_image.float(),
            'obs_seg_mask': obs_seg_mask_one_hot, # Provide the one-hot encoded mask
            'actions': actions_torch,
            'distance': torch.as_tensor(distance, dtype=torch.int64),
            'goal_pos': torch.as_tensor(goal_pos, dtype=torch.float32),
            'action_mask': action_mask,
        }
