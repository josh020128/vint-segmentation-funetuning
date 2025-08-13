"""
Semantic Segmentation enhanced ViNT using FuSe-style fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Tuple, Dict

# Import your fusion modules
from vint_train.models.vint.seg_fusion_modules import CrossModalAttention, SimpleFusion

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

class SegmentationViNT(nn.Module):
    """ViNT enhanced with semantic segmentation for improved navigation"""
    
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: int = 8,
        learn_angle: bool = False,
        obs_encoder: str = "resnet18",
        obs_encoding_size: int = 512,
        late_fusion: bool = False,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        # Segmentation specific parameters
        num_seg_classes: int = 5,
        seg_model_type: str = "unet",
        seg_encoder: str = "resnet34",
        fusion_type: str = "cross_attention",
        freeze_vint_encoder: bool = True,
        seg_feature_dim: int = 256,
        use_semantic_goals: bool = True,
    ):
        super().__init__()
        
        # Store configuration
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.late_fusion = late_fusion
        self.freeze_vint_encoder = freeze_vint_encoder
        self.use_semantic_goals = use_semantic_goals
        self.num_seg_classes = num_seg_classes
        self.obs_encoding_size = obs_encoding_size
        
        # Initialize base ViNT model
        self.vint_model = build_vint_model(
            context_size=context_size,
            len_traj_pred=len_traj_pred,
            learn_angle=learn_angle,
            obs_encoder=obs_encoder,
            obs_encoding_size=obs_encoding_size,
            late_fusion=late_fusion,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        
        # Freeze ViNT encoder initially (FuSe approach)
        if freeze_vint_encoder:
            for name, param in self.vint_model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
        
        # Initialize segmentation model with proper configuration
        if seg_model_type == "unet":
            self.seg_model = smp.Unet(
                encoder_name=seg_encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_seg_classes,
                activation=None,
            )
        elif seg_model_type == "deeplabv3plus":
            self.seg_model = smp.DeepLabV3Plus(
                encoder_name=seg_encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_seg_classes,
                activation=None,
            )
        elif seg_model_type == "fpn":
            self.seg_model = smp.FPN(
                encoder_name=seg_encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_seg_classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unknown segmentation model type: {seg_model_type}")
        
        # Segmentation feature extractor
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
        
        # Fusion module
        if fusion_type == "cross_attention":
            self.fusion_module = CrossModalAttention(
                embed_dim=obs_encoding_size,
                num_heads=mha_num_attention_heads
            )
        else:
            self.fusion_module = SimpleFusion(embed_dim=obs_encoding_size)
        
        # Action and distance predictors
        action_input_dim = obs_encoding_size * 2 if late_fusion else obs_encoding_size
        
        self.action_predictor = nn.Sequential(
            nn.Linear(action_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, len_traj_pred * 2),  # 2 for (x, y)
        )
        
        self.dist_predictor = nn.Sequential(
            nn.Linear(obs_encoding_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    
    def unfreeze_vint_encoder(self):
        """Unfreeze ViNT encoder for fine-tuning"""
        for param in self.vint_model.parameters():
            param.requires_grad = True
        print("Unfroze ViNT encoder parameters")
    
    def forward(
        self,
        obs_images: torch.Tensor,
        goal_images: torch.Tensor,
        obs_seg_masks: Optional[torch.Tensor] = None,  # Only for training supervision
        goal_seg_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - generates segmentation internally from RGB
        """
        batch_size = obs_images.shape[0]
        
        # Extract last observation frame for segmentation
        last_obs_frame = obs_images[:, -3:, :, :]  # Last RGB frame
        
        # Generate semantic segmentation from RGB
        obs_seg_logits = self.seg_model(last_obs_frame)
        
        # Generate goal segmentation if needed
        goal_seg_logits = None
        if self.use_semantic_goals:
            goal_seg_logits = self.seg_model(goal_images)
        
        # Extract segmentation features
        seg_features_obs = self.seg_feature_extractor(obs_seg_logits)
        
        # Get ViNT features
        with torch.set_grad_enabled(not self.freeze_vint_encoder):
            obs_encoding = self.vint_model.obs_encoder(obs_images)
            if hasattr(obs_encoding, 'shape') and len(obs_encoding.shape) > 2:
                obs_encoding = obs_encoding.view(batch_size, -1)
            obs_encoding = self.vint_model.compress_obs_enc(obs_encoding)
            
            goal_encoding = self.vint_model.goal_encoder(goal_images)
            if hasattr(goal_encoding, 'shape') and len(goal_encoding.shape) > 2:
                goal_encoding = goal_encoding.view(batch_size, -1)
            goal_encoding = self.vint_model.compress_goal_enc(goal_encoding)
        
        # Fuse features
        fused_obs_features = self.fusion_module(obs_encoding, seg_features_obs)
        
        # Distance prediction
        combined_features = torch.cat([fused_obs_features, goal_encoding], dim=1)
        dist_pred = self.dist_predictor(combined_features)
        
        # Action prediction
        if self.late_fusion:
            action_features = combined_features
        else:
            action_features = fused_obs_features
                        
        action_pred = self.action_predictor(action_features)
        action_pred = action_pred.view(batch_size, self.len_traj_pred, 2)
        
        return {
            'dist_pred': dist_pred,
            'action_pred': action_pred,
            'obs_seg_logits': obs_seg_logits,
            'goal_seg_logits': goal_seg_logits,
        }