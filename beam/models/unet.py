"""
U-Net architecture components for the diffusion model.

This module contains the building blocks of the U-Net architecture used
in the conditional diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AttentionFFN(nn.Module):
    def __init__(self, dim, hidden_dim,):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim*2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # x: (B, seq_len, dim)
        #GeGLU
        data, gate = self.linear1(x).chunk(2, dim=-1) # (B, seq_len, hidden_dim) for data and gate
        x = data * self.gelu(gate) # (B, seq_len, hidden_dim)

        x = self.linear2(x) # (B, seq_len, dim)
        return x
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.net = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        return self.net(x, x, x, need_weights=False)[0]

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=128, heads=4):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.heads = heads
        self.net = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, kdim=context_dim, vdim=context_dim, batch_first=True)

    def forward(self, x, context):
        return self.net(x, context, context, need_weights=False)[0]

class AttentionBlock(nn.Module):
    def __init__(self, dim, context_dim, mlp_dim, heads=4):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.mlp_dim = mlp_dim
        self.heads = heads  

        self.conv1 = nn.Conv2d(dim, dim, stride=1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(dim, dim, stride=1, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.layernorm3 = nn.LayerNorm(dim)
        self.groupnorm = nn.GroupNorm(32, dim)

        self.self_attn = SelfAttention(dim, heads)
        self.cross_attn = CrossAttention(dim, context_dim, heads)
        self.ffn = AttentionFFN(dim, mlp_dim)
    
    def forward(self, x, context):
        # Process input: B, C, H, W -> B, C, H, W 
        skip_x = x #long skip connection
        x = self.groupnorm(x)
        x = self.conv1(x) # shuffles features before heads
        B, C, H, W = x.shape
        
        # Self Attention Block B, C, H, W -> B, (H*W), C = B, Num_tokens, Embedding_dim -> B, Num_tokens, C
        x = rearrange(x, 'b c h w -> b (h w) c')
        attn_skip = x #short skip connection
        x = self.layernorm1(x)
        x = self.self_attn(x)
        x = x + attn_skip

        # Cross Attention Block B, Num_tokens, C -> B, Num_tokens, C
        cross_skip = x #short skip connection
        x = self.layernorm2(x)
        x = self.cross_attn(x, context)
        x = x + cross_skip

        # GeGLU Block B, Num_tokens, C -> B, Num_tokens, C -> B, C, H, W
        x = self.layernorm3(x)
        x = self.ffn(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.conv2(x)
        x = x + skip_x

        return x



class ResidualConvBlock(nn.Module):
    """
    Standard ResNet-style convolutional block with residual connections.
    """
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_res:
            return self.conv2(self.conv1(x))
            
        # Handle residual connection
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        # Add correct residual connection
        if self.same_channels:
            out = x + x2
        else:
            out = x1 + x2
            
        # Normalize by sqrt(2) to maintain variance
        return out / 1.414


class UnetDown(nn.Module):
    """
    Downsampling block for U-Net architecture.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels), 
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    """
    Upsampling block for U-Net architecture.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)  # Concatenate skip connection
        return self.model(x)


class EmbedFC(nn.Module):
    """
    Fully connected embedding network.
    """
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    """
    U-Net architecture with context embeddings for conditional diffusion.
    """
    def __init__(self, in_channels: int, in_dim: int, n_feat: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        # Initial convolution
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        
        # Downsampling path
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        # Bottleneck
        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.GELU())

        # Time and context embeddings
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(in_dim, 2 * n_feat)
        self.contextembed2 = EmbedFC(in_dim, n_feat)

        # Initial upsampling
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        
        # Upsampling path
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        
        # Final convolution
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input image (noisy)
            c: Context conditioning
            t: Diffusion Timestep
            context_mask: Mask for classifier-free guidance
            
        Returns:
            Predicted noise
        """
        # Process input through initial convolution and downsample
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hidden = self.to_vec(down2)

        # Reshape context for processing
        c = c.reshape((c.shape[0], c.shape[2]))
        # Apply context mask for classifier-free guidance
        context_mask = context_mask.reshape((x.shape[0], c.shape[1]))
        context_mask = -1 * (1 - context_mask)  # Flip 0 <-> 1
        c = c * context_mask

        # Create embeddings for context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # Upsampling path with skip connections
        up1 = self.up0(hidden)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # Add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        
        # Final convolution with skip connection to initial input
        out = self.out(torch.cat((up3, x), 1))
        return out
