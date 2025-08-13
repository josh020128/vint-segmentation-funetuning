"""
Advanced fusion modules for combining vision and segmentation features
"""
import torch
import torch.nn as nn
from typing import Dict

class SimpleFusion(nn.Module):
    """A simple but effective fusion module using concatenation and an MLP."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, vision_features: torch.Tensor, seg_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([vision_features, seg_features], dim=-1)
        fused = self.fusion_layer(combined)
        return fused

class CrossModalAttention(nn.Module):
    """
    A more powerful fusion that uses cross-attention.
    The vision feature "queries" the segmentation feature for relevant information.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1),
        )
        self.gate = nn.Parameter(torch.tensor([0.0])) # Learnable gate

    def forward(self, vision_features: torch.Tensor, seg_features: torch.Tensor) -> torch.Tensor:
        # Reshape for MHA: (B, L, E) where L is sequence length
        query = vision_features.unsqueeze(1) # (B, 1, E)
        key = seg_features.unsqueeze(1)      # (B, 1, E)
        value = key

        attended_features, _ = self.cross_attention(query=query, key=key, value=value)
        
        # Add & Norm (like a Transformer block)
        attended_features = self.norm1(query + attended_features)
        
        # FFN
        mlp_out = self.mlp(attended_features)
        
        # Add & Norm
        fused_features = self.norm2(attended_features + mlp_out)
        
        # Squeeze out the sequence length dimension
        fused_features = fused_features.squeeze(1)
        
        # Gated residual connection to the original vision feature
        final_features = torch.sigmoid(self.gate) * fused_features + (1 - torch.sigmoid(self.gate)) * vision_features

        return final_features

class FusionModule:
    @staticmethod
    def create(fusion_type: str, embed_dim: int = 512, **kwargs):
        if fusion_type == "simple":
            return SimpleFusion(embed_dim)
        elif fusion_type == "cross_attention":
            # Pass through any extra kwargs like num_heads
            return CrossModalAttention(embed_dim, **kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")