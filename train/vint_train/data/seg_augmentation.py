"""
Augmentation utilities for segmentation-enhanced navigation
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, Optional

class SynchronizedTransform:
    """Apply same transform to image and segmentation mask"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        augment_prob: float = 0.5,
    ):
        self.image_size = image_size
        self.augment_prob = augment_prob
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random rotation
        if torch.rand(1) < self.augment_prob:
            angle = torch.randint(-15, 15, (1,)).item()
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        
        # Random crop and resize
        if torch.rand(1) < self.augment_prob:
            crop_size = int(min(image.shape[-2:]) * np.random.uniform(0.7, 0.9))
            i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
            
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            
            image = TF.resize(image, self.image_size)
            mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        
        # Color augmentation (only for image)
        if torch.rand(1) < self.augment_prob:
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            image = jitter(image)
        
        return image, mask


class MixUp:
    """MixUp augmentation for segmentation"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(
        self,
        image1: torch.Tensor,
        mask1: torch.Tensor,
        image2: torch.Tensor,
        mask2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # For masks, use the one with higher lambda
        mixed_mask = mask1 if lam > 0.5 else mask2
        
        return mixed_image, mixed_mask