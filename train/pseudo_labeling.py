"""
Generate pseudo segmentation labels for navigation datasets
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Optional, Tuple
import segmentation_models_pytorch as smp

class PseudoLabelGenerator:
    """Generate pseudo segmentation labels using pre-trained models"""
    
    def __init__(
        self,
        model_name: str = "deeplabv3plus",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        num_classes: int = 19,  # Cityscapes classes
        navigation_classes: dict = None,
    ):
        self.device = torch.device(device)
        
        # Load pre-trained segmentation model
        if checkpoint_path:
            self.model = torch.load(checkpoint_path)
        else:
            # Use a pre-trained model from segmentation_models_pytorch
            self.model = smp.DeepLabV3Plus(
                encoder_name="resnet101",
                encoder_weights="imagenet",
                classes=num_classes,
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Mapping from original classes to navigation classes
        self.class_mapping = navigation_classes or {
            # Cityscapes to navigation mapping
            0: 0,   # road -> walkable
            1: 0,   # sidewalk -> walkable
            2: 1,   # building -> obstacle
            3: 1,   # wall -> obstacle
            4: 1,   # fence -> obstacle
            5: 2,   # pole -> vertical_obstacle
            6: 2,   # traffic_light -> vertical_obstacle
            7: 2,   # traffic_sign -> vertical_obstacle
            8: 3,   # vegetation -> soft_obstacle
            9: 4,   # terrain -> off_road
            10: 4,  # sky -> off_road
            11: 5,  # person -> dynamic
            12: 5,  # rider -> dynamic
            13: 6,  # car -> vehicle
            14: 6,  # truck -> vehicle
            15: 6,  # bus -> vehicle
            16: 6,  # train -> vehicle
            17: 6,  # motorcycle -> vehicle
            18: 6,  # bicycle -> vehicle
        }
    
    def generate_labels(
        self,
        image_path: str,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Generate pseudo label for a single image"""
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess_image(image)
        
        # Generate segmentation
        with torch.no_grad():
            seg_logits = self.model(image_tensor.to(self.device))
            seg_pred = torch.argmax(seg_logits, dim=1).cpu().numpy()[0]
        
        # Map to navigation classes
        lookup_table = np.zeros(256, dtype=np.uint8)
        for orig_class, nav_class in self.class_mapping.items():
            lookup_table[orig_class] = nav_class
            
        # Apply the mapping in a single, fast operation
        nav_mask = lookup_table[seg_pred]

        for orig_class, nav_class in self.class_mapping.items():
            nav_mask[seg_pred == orig_class] = nav_class
        
        # Save if path provided
        if save_path:
            mask_image = Image.fromarray(nav_mask.astype(np.uint8))
            mask_image.save(save_path)
        
        return nav_mask
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for segmentation model"""
        # Resize and normalize
        image = image.resize((512, 512))
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        return image_tensor.unsqueeze(0).float()


class HeuristicLabelGenerator:
    """Generate heuristic segmentation labels based on simple rules"""
    
    def __init__(self, image_size: Tuple[int, int] = (120, 160)):
        self.image_size = image_size
    
    def generate_labels(
        self,
        image: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate heuristic labels based on image features"""
        
        h, w = self.image_size
        mask = torch.zeros((h, w), dtype=torch.long)
        
        # Simple heuristics:
        # 1. Bottom portion is likely walkable
        mask[int(h * 0.6):, :] = 0  # Walkable
        
        # 2. Use color/texture for obstacle detection
        if image.shape[0] >= 3:  # RGB image
            # Green areas might be vegetation
            green_channel = image[1]
            vegetation_mask = green_channel > image[0] + 0.1
            mask[vegetation_mask] = 3  # Vegetation
        
        # 3. Use depth if available
        if depth is not None:
            # Close objects might be obstacles
            close_mask = depth < 0.3  # Normalized depth
            mask[close_mask] = 1  # Obstacle
        
        return mask