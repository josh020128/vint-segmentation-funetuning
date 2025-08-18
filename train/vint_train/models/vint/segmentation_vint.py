# FILE: vint_train/models/vint/segmentation_vint.py

"""
Semantic Segmentation enhanced ViNT using a stable, late-fusion architecture.
This is the recommended approach for stability and effective transfer learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict
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

# my current code using lated fusion. Not good seg_loss
class SegmentationViNT(nn.Module):
    """
    ViNT enhanced with a semantic segmentation branch using late fusion for stability.
    """
    def __init__(
        self,
        # ViNT-specific arguments
        context_size: int,
        len_traj_pred: int,
        learn_angle: bool,
        obs_encoder: str,
        obs_encoding_size: int,
        
        # Segmentation-specific arguments
        num_seg_classes: int,
        seg_encoder: str = "resnet34",
        freeze_vint: bool = True,
        seg_feature_dim: int = 256,
        **kwargs # Absorb any extra unused parameters
    ):
        super().__init__()
        
        # Store key configuration parameters
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.num_seg_classes = num_seg_classes
        
        # 1. Initialize the base ViNT model
        # The full ViNT model will be used as a feature extractor
        vint_args = {
            'context_size': context_size, 'len_traj_pred': len_traj_pred,
            'learn_angle': learn_angle, 'obs_encoder': obs_encoder,
            'obs_encoding_size': obs_encoding_size,
            **kwargs
        }
        self.vint_model = build_vint_model(**vint_args)
        
        # Freeze the entire ViNT model if specified
        if freeze_vint:
            print("Freezing all ViNT model parameters.")
            for param in self.vint_model.parameters():
                param.requires_grad = False
        
        # 2. Initialize the Segmentation Branch
        self.seg_model = smp.Unet(
            encoder_name=seg_encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_seg_classes,
            activation=None,
        )
        
        # 3. Initialize a Segmentation Feature Extractor
        # This module processes segmentation logits into a 1D feature vector
        self.seg_feature_extractor = nn.Sequential(
            nn.Conv2d(num_seg_classes, seg_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(seg_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(seg_feature_dim * 49, obs_encoding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        # 4. Initialize NEW Prediction Heads for the fused features
        # The input dimension is doubled because we concatenate the two feature vectors
        fused_dim = obs_encoding_size * 2
        num_action_outputs = len_traj_pred * (3 if learn_angle else 2)
        
        self.action_predictor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_action_outputs),
        )
        
        # The distance predictor takes the fused observation and the goal
        dist_predictor_input_dim = fused_dim + obs_encoding_size
        self.dist_predictor = nn.Sequential(
            nn.Linear(dist_predictor_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def unfreeze_vint(self):
        """Unfreeze all ViNT parameters for Stage 2 finetuning."""
        print("Unfreezing ViNT model parameters...")
        for param in self.vint_model.parameters():
            param.requires_grad = True
    
    def forward(self, obs_images: torch.Tensor, goal_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = obs_images.shape[0]
        
        # --- Branch 1: Get ViNT Features ---
        # We get the final 1D feature vectors from the pre-trained model
        with torch.set_grad_enabled(next(self.vint_model.parameters()).requires_grad):
            obs_encoding = self.vint_model.compress_obs_enc(self.vint_model.obs_encoder(obs_images))
            goal_encoding = self.vint_model.compress_goal_enc(self.vint_model.goal_encoder(goal_images))

            # ADDED CODE 1
            # Normalize for stability
            obs_encoding = F.normalize(obs_encoding, p=2, dim=-1) * 10
            goal_encoding = F.normalize(goal_encoding, p=2, dim=-1) * 10
        
        # --- Branch 2: Get Segmentation Features ---
        last_obs_frame = obs_images[:, -3:, :, :]

        obs_seg_logits = self.seg_model(last_obs_frame)

        seg_probs = F.softmax(obs_seg_logits, dim=1).detach()
        seg_features_obs = self.seg_feature_extractor(seg_probs)
        seg_features_obs = F.normalize(seg_features_obs, p=2, dim=-1) * 10
        
        # --- Late Fusion Step ---
        # Concatenate the final 1D feature vectors from both branches
        fused_obs_features = torch.cat([obs_encoding, seg_features_obs], dim=1)
        
        # --- Final Prediction ---
        # The new prediction heads take the fused observation features and the original goal features
        
        # For simplicity, we assume the action predictor takes the fused observation
        # and the distance predictor takes the combined features.
        pred_actions = self.action_predictor(fused_obs_features)
        
        combined_features = torch.cat([fused_obs_features, goal_encoding], dim=1)
        pred_dist = self.dist_predictor(combined_features)

        return {
            'dist_pred': pred_dist,
            'action_pred': pred_actions.view(batch_size, self.len_traj_pred, -1),
            'obs_seg_logits': obs_seg_logits,
        }
