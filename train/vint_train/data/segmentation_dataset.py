"""
Dataset that loads navigation data with semantic segmentation
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Dict
import os

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.data_utils import get_data_path, calculate_sin_cos

class ViNTSegmentationDataset(ViNT_Dataset):
    """Extended ViNT dataset with semantic segmentation masks"""
    
    def __init__(
        self,
        *args,
        is_train: bool = True, 
        seg_data_folder: Optional[str] = None,
        seg_model_name: str = "scand",
        use_pseudo_labels: bool = False,
        pseudo_label_model: Optional[str] = None,
        seg_augmentation_prob: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.is_train = is_train    
        self.seg_data_folder = seg_data_folder
        self.seg_model_name = seg_model_name
        self.use_pseudo_labels = use_pseudo_labels
        self.seg_augmentation_prob = seg_augmentation_prob
        
        # Setup segmentation classes
        self.setup_segmentation_classes()
        
        # Log configuration
        print(f"\n{'='*60}")
        print("ViNTSegmentationDataset Configuration:")
        print(f"  - Dataset: {kwargs.get('dataset_name', 'unknown')}")
        print(f"  - seg_data_folder: {seg_data_folder}")
        print(f"  - use_pseudo_labels: {use_pseudo_labels}")
        print(f"  - seg_model_name: {seg_model_name}")
        print(f"  - num_classes: {self.num_classes}")
        print(f"  - class_names: {self.class_names}")
        
        # Test loading a segmentation mask
        if len(self.index_to_data) > 0:
            self._test_segmentation_loading()
        print(f"{'='*60}\n")
    
    def train(self):
        """Set the dataset to training mode (for augmentations)."""
        self.is_train = True

    def eval(self):
        """Set the dataset to evaluation mode (no augmentations)."""
        self.is_train = False

    def setup_segmentation_classes(self):
        """Define segmentation classes based on model/dataset"""
        if self.seg_model_name == "scand":
            self.num_classes = 5
            self.class_names = ['floor', 'wall', 'door', 'furniture', 'unknown']
        elif self.seg_model_name == "cityscapes":
            self.num_classes = 8
            self.class_names = ['walkable', 'obstacle', 'vertical', 'vegetation', 
                               'terrain', 'dynamic', 'vehicle', 'unknown']
        else:
            self.num_classes = 3
            self.class_names = ['navigable', 'obstacle', 'unknown']
    
    def get_seg_mask_path(self, image_path: str) -> str:
        """Convert image path to segmentation mask path"""
        # Get the relative path structure
        rel_path = os.path.relpath(image_path, self.data_folder)
        
        # Replace extension with _seg.png
        dir_path = os.path.dirname(rel_path)
        base_name = os.path.basename(rel_path)
        name_without_ext = os.path.splitext(base_name)[0]
        seg_filename = f"{name_without_ext}_seg.png"
        
        # Build full segmentation path
        if dir_path:
            seg_path = os.path.join(self.seg_data_folder, dir_path, seg_filename)
        else:
            seg_path = os.path.join(self.seg_data_folder, seg_filename)
        
        return seg_path
    
    def load_segmentation_mask(self, image_path: str) -> torch.Tensor:
        """Load or generate segmentation mask for an image"""
        
        # Try to load existing segmentation mask if folder is provided
        if self.seg_data_folder:
            seg_path = self.get_seg_mask_path(image_path)
            if os.path.exists(seg_path):
                return self.load_mask_from_file(seg_path)
            elif not self.use_pseudo_labels:
                # Only warn once per run to avoid spam
                if not hasattr(self, '_warned_missing'):
                    self._warned_missing = True
                    print(f"Warning: Segmentation masks not found, using heuristics")
        
        # Use heuristic if pseudo labels are enabled or as fallback
        if self.use_pseudo_labels or self.seg_data_folder is None:
            return self.generate_heuristic_mask()
        
        # Return zeros if nothing works
        return torch.zeros(self.image_size, dtype=torch.long)
    
    def load_mask_from_file(self, mask_path: str) -> torch.Tensor:
        """Load segmentation mask from file"""
        try:
            seg_mask = Image.open(mask_path)
            
            # Convert to grayscale if needed
            if seg_mask.mode != 'L':
                seg_mask = seg_mask.convert('L')
            
            # Resize to match image size (PIL expects width, height)
            seg_mask = seg_mask.resize(
                (self.image_size[1], self.image_size[0]),
                Image.NEAREST  # Use nearest neighbor for masks
            )
            
            # Convert to numpy then tensor
            seg_array = np.array(seg_mask)
            
            # Convert to tensor
            seg_tensor = torch.tensor(seg_array, dtype=torch.long)
            
            # Clamp to valid range
            seg_tensor = torch.clamp(seg_tensor, 0, self.num_classes - 1)
            
            return seg_tensor
            
        except Exception as e:
            print(f"Error loading mask from {mask_path}: {e}")
            return self.generate_heuristic_mask()
    
    def generate_heuristic_mask(self) -> torch.Tensor:
        """Generate simple heuristic segmentation"""
        h, w = self.image_size
        
        # Start with unknown
        mask = torch.ones((h, w), dtype=torch.long) * (self.num_classes - 1)
        
        # Bottom 40% is floor
        floor_start = int(h * 0.6)
        mask[floor_start:, :] = 0  # Floor
        
        # Top 30% is wall
        mask[:int(h * 0.3), :] = 1  # Wall
        
        # Add random furniture during training
        # Use self.training from parent class (set by train()/eval() methods)
        if self.is_train and np.random.random() < self.seg_augmentation_prob:
            num_obstacles = np.random.randint(1, 3)
            for _ in range(num_obstacles):
                x = np.random.randint(w // 4, 3 * w // 4)
                y = np.random.randint(int(h * 0.4), int(h * 0.7))
                w_obs = np.random.randint(10, min(20, w - x))
                h_obs = np.random.randint(10, min(20, h - y))
                mask[y:y+h_obs, x:x+w_obs] = 3  # Furniture
        
        return mask
    
    def _test_segmentation_loading(self):
        """Test if segmentation loading works"""
        print("\nTesting segmentation loading...")
        
        # Get first sample
        f_curr, curr_time, _ = self.index_to_data[0]
        test_path = get_data_path(self.data_folder, f_curr, curr_time)
        
        print(f"  Test image path: {test_path}")
        
        if self.seg_data_folder:
            seg_path = self.get_seg_mask_path(test_path)
            print(f"  Expected seg path: {seg_path}")
            print(f"  Seg file exists: {os.path.exists(seg_path)}")
        
        # Try loading
        seg_mask = self.load_segmentation_mask(test_path)
        print(f"  Loaded seg shape: {seg_mask.shape}")
        unique_vals = torch.unique(seg_mask).tolist()
        print(f"  Unique values: {unique_vals}")
        
        # Check distribution
        total_pixels = seg_mask.numel()
        for i in range(self.num_classes):
            count = (seg_mask == i).sum().item()
            if count > 0:
                percentage = count / total_pixels * 100
                print(f"    Class {i} ({self.class_names[i]}): {percentage:.1f}%")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with segmentation masks"""
        # Get base data from parent
        f_curr, curr_time, max_goal_dist = self.index_to_data[idx]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)
        
        # Load observation images (context stack) using parent's method
        context_times = list(range(
            curr_time - (self.context_size - 1) * self.waypoint_spacing,
            curr_time + 1,
            self.waypoint_spacing
        ))
        
        obs_images_list = []
        for t in context_times:
            img_path = get_data_path(self.data_folder, f_curr, t)
            # Use parent's _load_image_and_resize method
            obs_images_list.append(self._load_image_and_resize(img_path))
        obs_images = torch.cat(obs_images_list)  # Shape: (3*context_size, H, W)
        
        # Load goal image using parent's method
        goal_path = get_data_path(self.data_folder, f_goal, goal_time)
        goal_image = self._load_image_and_resize(goal_path)  # Shape: (3, H, W)
        
        # Load segmentation masks for current observation and goal
        obs_seg_path = get_data_path(self.data_folder, f_curr, curr_time)
        goal_seg_path = get_data_path(self.data_folder, f_goal, goal_time)
        
        obs_seg_mask = self.load_segmentation_mask(obs_seg_path)  # Shape: (H, W)
        goal_seg_mask = self.load_segmentation_mask(goal_seg_path)  # Shape: (H, W)
        
        # Compute navigation data using parent's methods
        curr_traj_data = self._get_trajectory(f_curr)
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        # Calculate distance
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
        
        # Process actions
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        
        # Create action mask
        action_mask = torch.tensor(
            (distance < self.max_action_distance) and 
            (distance > self.min_action_distance) and 
            (not goal_is_negative),
            dtype=torch.float32
        )
        
        return {
            'obs_images': obs_images.float(),
            'goal_images': goal_image.float(),
            'obs_seg_mask': obs_seg_mask,
            'goal_seg_mask': goal_seg_mask,
            'actions': actions_torch,
            'distance': torch.as_tensor(distance, dtype=torch.int64),
            'goal_pos': torch.as_tensor(goal_pos, dtype=torch.float32),
            'dataset_idx': torch.as_tensor(self.dataset_index, dtype=torch.int64),
            'action_mask': action_mask,
        }