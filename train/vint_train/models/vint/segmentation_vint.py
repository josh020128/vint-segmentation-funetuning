"""
Semantic Segmentation enhanced ViNT using FuSe-style fusion
Modified with Gated Fusion for stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Tuple, Dict

# Import your fusion modules - we'll add GatedFusion here
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

class GatedFusion(nn.Module):
    """
    Stable gated fusion that uses segmentation to guide navigation features.
    Specifically designed to encourage floor/navigable area attention.
    """
    def __init__(self, embed_dim: int = 512, num_seg_classes: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_seg_classes = num_seg_classes

        # !! Add LayerNorm to normalize features before fusion
        self.vision_norm = nn.LayerNorm(embed_dim)
        self.seg_norm = nn.LayerNorm(embed_dim)
        
        # Convert segmentation features to attention weights
        # We use a small network to learn which segmentation classes should enhance features
        self.seg_to_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()  # Bounded between [0, 1] for stability
        )
        
        # Learnable gate to control fusion strength (starts at 0.5)
        self.fusion_strength = nn.Parameter(torch.tensor(0.5))
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.8))
        
    def forward(self, vision_features: torch.Tensor, seg_features: torch.Tensor) -> torch.Tensor:
        """
        Apply gated fusion with emphasis on navigable areas.
        
        Args:
            vision_features: (B, D) - ViNT encoded features
            seg_features: (B, D) - Segmentation encoded features
        
        Returns:
            fused_features: (B, D) - Gated combination
        """

        # !!CRITICAL: Normalize both features to same scale
        vision_features_norm = self.vision_norm(vision_features)
        seg_features_norm = self.seg_norm(seg_features)

        # !!Additional safety: clamp normalized features
        vision_features_norm = torch.clamp(vision_features_norm, -10, 10)
        seg_features_norm = torch.clamp(seg_features_norm, -10, 10)

        # Generate attention gate from segmentation features --> changed to norm
        gate = self.seg_to_gate(seg_features_norm)
        
        # Apply gate with learnable fusion strength
        fusion_strength = torch.sigmoid(self.fusion_strength)
        
        # Gated fusion with strong residual connection for stability
        # This ensures we always keep most of the original ViNT features
        residual_weight = torch.sigmoid(self.residual_weight)
        
        # Combine: keep most of original, add gated segmentation guidance --> changed to norm
        fused_features = residual_weight * vision_features_norm + \
                        (1 - residual_weight) * fusion_strength * gate * vision_features_norm
        
        # !! Final safety clamp
        fused_features = torch.clamp(fused_features, -10, 10)

        return fused_features

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
        self.fusion_type = fusion_type  # Store fusion type
        
        # Initialize base ViNT model (unchanged)
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
        
        # Freeze ViNT encoder initially (unchanged)
        if freeze_vint_encoder:
            for name, param in self.vint_model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
        
        # Initialize segmentation model (unchanged)
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
        
        # Segmentation feature extractor (unchanged)
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
        
        # Fusion module - now supports gated fusion
        if fusion_type == "gated":
            self.fusion_module = GatedFusion(
                embed_dim=obs_encoding_size,
                num_seg_classes=num_seg_classes
            )
        elif fusion_type == "cross_attention":
            self.fusion_module = CrossModalAttention(
                embed_dim=obs_encoding_size,
                num_heads=mha_num_attention_heads
            )
        else:
            self.fusion_module = SimpleFusion(embed_dim=obs_encoding_size)
        
        # Action and distance predictors (unchanged)
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
    
    # <<< ADDED: Helper function for NaN debugging >>>
    def _check_nan(self, tensor: torch.Tensor, name: str):
        """Checks if a tensor contains NaN or Inf values and prints a warning."""
        if not torch.isfinite(tensor).all():
            print(f"⚠️ WARNING: NaN or Inf detected in tensor '{name}'")

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
        
        # Extract last observation frame for segmentation (unchanged)
        last_obs_frame = obs_images[:, -3:, :, :]  # Last RGB frame
        
        # Generate semantic segmentation from RGB (unchanged)
        obs_seg_logits = self.seg_model(last_obs_frame)
        self._check_nan(obs_seg_logits, "obs_seg_logits") # Debugging code 1
        
        # Add stability clamping for gated fusion
        if self.fusion_type == "gated":
            obs_seg_logits = torch.clamp(obs_seg_logits, -10, 10)
        
        # Generate goal segmentation if needed (unchanged)
        goal_seg_logits = None
        if self.use_semantic_goals:
            goal_seg_logits = self.seg_model(goal_images)
            if self.fusion_type == "gated":
                goal_seg_logits = torch.clamp(goal_seg_logits, -10, 10)
        
        # Extract segmentation features (unchanged)
        seg_features_obs = self.seg_feature_extractor(obs_seg_logits)
        self._check_nan(seg_features_obs, "seg_features_obs") # Debugging code 2
        
        # Get ViNT features (unchanged)
        with torch.set_grad_enabled(not self.freeze_vint_encoder):
            obs_encoding = self.vint_model.obs_encoder(obs_images)
            if hasattr(obs_encoding, 'shape') and len(obs_encoding.shape) > 2:
                obs_encoding = obs_encoding.view(batch_size, -1)
            obs_encoding = self.vint_model.compress_obs_enc(obs_encoding)
            self._check_nan(obs_encoding, "obs_encoding") # Debugging code 3

            # !! CRITICAL: Normalize ViNT features before fusion
            # This ensures both feature streams have similar magnitudes
            obs_encoding = F.normalize(obs_encoding, p=2, dim=-1) * 10  # L2 norm then scale
            
            goal_encoding = self.vint_model.goal_encoder(goal_images)
            if hasattr(goal_encoding, 'shape') and len(goal_encoding.shape) > 2:
                goal_encoding = goal_encoding.view(batch_size, -1)
            goal_encoding = self.vint_model.compress_goal_enc(goal_encoding)
            self._check_nan(goal_encoding, "goal_encoding") # Debugging code 4

            # !! Normalize goal encoding too
            goal_encoding = F.normalize(goal_encoding, p=2, dim=-1) * 10
        
        # !! CRITICAL: Normalize segmentation features
        seg_features_obs = F.normalize(seg_features_obs, p=2, dim=-1) * 10

        # Add stability clamping before fusion
        if self.fusion_type == "gated":
            obs_encoding = torch.clamp(obs_encoding, -5, 5)
            seg_features_obs = torch.clamp(seg_features_obs, -5, 5)
        
        # Fuse features (using the selected fusion module)
        fused_obs_features = self.fusion_module(obs_encoding, seg_features_obs)
        self._check_nan(fused_obs_features, "fused_obs_features") # Debugging code 5
        
        # Add stability clamping after fusion
        if self.fusion_type == "gated":
            fused_obs_features = torch.clamp(fused_obs_features, -10, 10)
        
        # Distance prediction (unchanged)
        combined_features = torch.cat([fused_obs_features, goal_encoding], dim=1)
        dist_pred = self.dist_predictor(combined_features)
        self._check_nan(dist_pred, "dist_pred") # Debugging code 6
        
        # Action prediction (unchanged)
        if self.late_fusion:
            action_features = combined_features
        else:
            action_features = fused_obs_features
                        
        action_pred = self.action_predictor(action_features)
        self._check_nan(action_pred, "action_pred") # Debugging code 7
        action_pred = action_pred.view(batch_size, self.len_traj_pred, 2)
        
        return {
            'dist_pred': dist_pred,
            'action_pred': action_pred,
            'obs_seg_logits': obs_seg_logits,
            'goal_seg_logits': goal_seg_logits,
        }