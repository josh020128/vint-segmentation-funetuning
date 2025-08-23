# FILE: vint_train/models/vint/vint.py

"""
Core Visual Navigation Transformer (ViNT) model.
This version has been updated to support ResNet backbones, making it
compatible with the dual-input project's configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import torchvision.models as models

# Assuming these helper modules exist in your project structure
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder

def create_encoder(encoder_name: str, in_channels: int, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Creates a visual encoder (e.g., ResNet) and modifies it for our task.
    
    Args:
        encoder_name (str): The name of the encoder (e.g., "resnet34").
        in_channels (int): The number of input channels for the first layer.
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        Tuple[nn.Module, int]: A tuple containing the encoder model and its
                               output feature dimension.
    """
    if "resnet" in encoder_name:
        if encoder_name == "resnet34":
            encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = encoder.fc.in_features
        else:
            raise NotImplementedError(f"ResNet variant '{encoder_name}' not supported.")

        # Modify the first convolutional layer to accept the correct number of input channels
        encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final classification layer with an identity layer to use as a feature extractor
        encoder.fc = nn.Identity()
        return encoder, num_features
    else:
        raise ValueError(f"Encoder '{encoder_name}' is not supported. Please use a ResNet variant.")


class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: int = 8,
        learn_angle: bool = False,
        obs_encoder: str = "resnet34",
        obs_encoding_size: int = 512,
        mha_num_attention_heads: int = 8,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        **kwargs # Absorb unused kwargs from the config
    ) -> None:
        super(ViNT, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size

        # --- Visual Encoders ---
        # Encoder for the full stack of context images (e.g., 5 frames * 3 channels = 15 channels)
        self.obs_encoder, num_obs_features = create_encoder(obs_encoder, in_channels=3 * (self.context_size + 1))
        
        # A separate 3-channel encoder for the single goal image
        self.goal_encoder, num_goal_features = create_encoder(obs_encoder, in_channels=3)

        # --- Compression Layers ---
        self.compress_obs_enc = nn.Linear(num_obs_features, self.obs_encoding_size) if num_obs_features != self.obs_encoding_size else nn.Identity()
        self.compress_goal_enc = nn.Linear(num_goal_features, self.obs_encoding_size) if num_goal_features != self.obs_encoding_size else nn.Identity()

        # --- Transformer Decoder ---
        # Note: The original ViNT used a transformer. For the dual-input model,
        # the wrapper handles the feature extraction, so this part is simplified.
        # This forward pass is primarily for getting the feature encodings.
        # If you were to run ViNT standalone, you would need the decoder.
        # For this architecture, we only need the encoders and compression layers.

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        This forward pass is simplified for the dual-input architecture.
        Its main purpose is to provide the obs_encoding and goal_encoding.
        The trajectory prediction is handled by the wrapper model.
        """
        # This is a placeholder for a full forward pass.
        # The dual-input wrapper will call the encoder components directly.
        # We return dummy values to satisfy the structure if this were called directly.
        batch_size = obs_img.shape[0]
        dummy_dist = torch.zeros((batch_size, 1), device=obs_img.device)
        dummy_actions = torch.zeros((batch_size, self.len_trajectory_pred, self.num_action_params), device=obs_img.device)
        
        return {
            'dist_pred': dummy_dist,
            'action_pred': dummy_actions,
        }
