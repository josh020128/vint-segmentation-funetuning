# FILE: vint_train/models/vint/segmentation_vint.py

"""
Semantic Segmentation enhanced ViNT using a stable, late-fusion architecture.
This is the recommended approach for stability and effective transfer learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict, Tuple
from prettytable import PrettyTable

# It's assumed you have a working ViNT implementation
from vint_train.models.vint.vint import ViNT

def build_vint_model(**kwargs) -> nn.Module:
    """Build a ViNT model or placeholder"""
    try:
        from vint_train.models.vint.vint import ViNT
        return ViNT(**kwargs)
    except:
        # Simplified placeholder for testing
        class SimpleViNT(nn.Module):
            def __init__(self, obs_encoding_size=512, context_size=5, **kwargs):
                super().__init__()
                self.obs_encoding_size = obs_encoding_size
                self.context_size = context_size
                
                # Simple encoders
                self.obs_encoder = nn.Sequential(
                    nn.Conv2d(3 * context_size, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                self.goal_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                self.compress_obs_enc = nn.Linear(64, obs_encoding_size)
                self.compress_goal_enc = nn.Linear(64, obs_encoding_size)
                
        return SimpleViNT(**kwargs)

def count_parameters(model):
    """Count trainable parameters"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params

class SpatialSegmentationEncoder(nn.Module):
    """
    Encodes a segmentation mask, preserving spatial information to identify
    obstacle locations and producing a feature summary.
    """
    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        
    def forward(self, seg_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(seg_masks)
        x = self.conv2(x)
        spatial_features = self.conv3(x) # Shape: (B, 128, H/8, W/8)
        
        # Create the feature summary vector
        features = self.feature_extractor(spatial_features)
        
        return features, spatial_features

class TrajectoryAdapter(nn.Module):
    """
    Generates corrective offsets for a trajectory based on spatial obstacle features.
    """
    def __init__(self, feature_dim: int, traj_len: int):
        super().__init__()
        # This network learns to generate small x,y nudges for each waypoint
        self.trajectory_modulator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, traj_len * 2),  # x,y offsets for each waypoint
            nn.Tanh() # Output values between -1 and 1
        )
        
    def forward(self, vint_traj: torch.Tensor, seg_features: torch.Tensor) -> torch.Tensor:
        B, T, _ = vint_traj.shape
        
        # Generate trajectory offsets based on the summary of the segmentation mask
        offsets = self.trajectory_modulator(seg_features)
        # Reshape and scale down the offsets to ensure they are small corrections
        offsets = offsets.view(B, T, 2) * 0.2  # Limit max offset to 20% of normalized space
        
        # Apply the corrective nudge to the original trajectory
        adapted_traj = vint_traj[:, :, :2] + offsets
        
        # Re-attach the angle information if it exists
        if vint_traj.shape[-1] > 2:
            adapted_traj = torch.cat([adapted_traj, vint_traj[:, :, 2:]], dim=-1)
            
        return adapted_traj

class SegmentationViNT(nn.Module):
    """
    The main model that orchestrates the "driver" (ViNT) and "co-pilot" (segmentation) modules.
    """
    def __init__(
        self,
        context_size: int, len_traj_pred: int, learn_angle: bool,
        obs_encoder: str, obs_encoding_size: int, num_seg_classes: int,
        seg_feature_dim: int = 256, freeze_vint: bool = True, **kwargs
    ):
        super().__init__()
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        
        # 1. ViNT model (the "driver")
        vint_args = {
            'context_size': context_size, 'len_traj_pred': len_traj_pred,
            'learn_angle': learn_angle, 'obs_encoder': obs_encoder,
            'obs_encoding_size': obs_encoding_size, **kwargs
        }
        self.vint_model = build_vint_model(**vint_args)
        
        if freeze_vint:
            print("Freezing ViNT model parameters for Stage 1.")
            for param in self.vint_model.parameters():
                param.requires_grad = False
        
        # 2. Segmentation processing modules (the "co-pilot")
        self.seg_encoder = SpatialSegmentationEncoder(num_seg_classes, seg_feature_dim)
        self.trajectory_adapter = TrajectoryAdapter(seg_feature_dim, len_traj_pred)
        
        # 3. Learnable parameter to control the influence of the co-pilot's corrections
        self.seg_influence_logit = nn.Parameter(torch.tensor(-1.38)) # sigmoid(-1.38) is approx 0.2
        
        # 4. Prediction heads
        # The distance predictor now uses both RGB and segmentation features for a more informed estimate.
        fused_dim = obs_encoding_size + seg_feature_dim
        self.dist_predictor = nn.Sequential(
            nn.Linear(fused_dim + obs_encoding_size, 256), # Fused Obs + Goal
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        print("Initialized SegmentationViNT (Co-Pilot Architecture)")
        count_parameters(self)
    
    def unfreeze_vint(self):
        print("Unfreezing ViNT model for Stage 2...")
        for param in self.vint_model.parameters():
            param.requires_grad = True
            
    def forward(self, 
                obs_images: torch.Tensor,
                goal_images: torch.Tensor,
                obs_seg_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # 1. Get ViNT's primary plan and features from the RGB input
        vint_outputs = self.vint_model(obs_images, goal_images)
        vint_trajectory = vint_outputs['action_pred']
        
        # Gradients are controlled by the `freeze_vint` flag, so this block is safe.
        obs_encoding = self.vint_model.compress_obs_enc(self.vint_model.obs_encoder(obs_images))
        goal_encoding = self.vint_model.compress_goal_enc(self.vint_model.goal_encoder(goal_images))
        
        # <<< NAN LOSS FIX: Normalize all feature vectors and apply a scaling factor >>>
        obs_encoding = F.normalize(obs_encoding, p=2, dim=-1) * 10
        goal_encoding = F.normalize(goal_encoding, p=2, dim=-1) * 10

        # 2. Process segmentation mask to get safety information
        seg_features, spatial_features = self.seg_encoder(obs_seg_masks)
        seg_features = F.normalize(seg_features, p=2, dim=-1) * 10
        
        # 3. The "co-pilot" adapts the trajectory based on obstacles
        adapted_trajectory = self.trajectory_adapter(vint_trajectory.detach(), seg_features)
        
        # 4. Blend the original and adapted trajectories with a learned, limited influence
        seg_influence = torch.sigmoid(self.seg_influence_logit)
        final_trajectory = torch.lerp(vint_trajectory, adapted_trajectory, seg_influence)
        
        # 5. Predict distance using features from both RGB and segmentation
        fused_obs_features = torch.cat([obs_encoding, seg_features], dim=1)
        combined_for_dist = torch.cat([fused_obs_features, goal_encoding], dim=1)
        dist_pred = self.dist_predictor(combined_for_dist)
        
        return {
            'action_pred': final_trajectory,
            'dist_pred': dist_pred,
            'vint_trajectory': vint_trajectory.detach(), # For consistency loss
            'seg_influence': seg_influence.item(),
        }
