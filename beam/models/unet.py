"""
U-Net architecture components for the diffusion model.

This module contains the building blocks of the U-Net architecture used
in the conditional diffusion model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (B, 1, 1, 1)
        Returns:
        - embeddings: (B, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)
    
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
        # print(f'context shape: {context.shape}')
        # print('\n')
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
        self, in_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.conv1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        
        self.conv2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if not self.same_channels else nn.Identity()

        nn.init.zeros_(self.conv2[-1].weight) # Start Learning from Identity: resnet(x) = x

    def forward(self, x: torch.Tensor, c: torch.Tensor, tin: torch.Tensor, tout: torch.Tensor) -> torch.Tensor:
        # x: B, in_channels, H, W
        # c: B, context_dim if it is the first block, else time
        # tin: B, in_channels*2
        # tout: B, out_channels*2
        
        # Handle residual connection
        xc = self.norm1(x)

        betat, gammat = tin.view(-1, 2*self.in_channels, 1, 1).chunk(2, dim=1)
        betac, gammac = c.view(-1, 2*self.in_channels, 1, 1).chunk(2, dim=1) if c is not None else torch.zeros_like(tin).view(-1, 2*self.in_channels, 1, 1).chunk(2, dim=1)
        # print(f'betac shape: {betac.shape}')
        # print(f'gammac shape: {gammac.shape}')
        # print(f'betat shape: {betat.shape}')
        # print(f'gammat shape: {gammat.shape}')
        # print(f'x shape: {xc.shape}')
        # print('\n')
        xc = xc*(betac + betat) + gammac + gammat #FiLM style embedding
        xc = self.conv1(xc)


        xc = self.norm2(xc)
        # print(f'xc2 shape: {xc.shape}')
        # print('\n')
        betat, gammat = tout.view(-1, 2*self.out_channels, 1, 1).chunk(2, dim=1)
        # print(f'betat2 shape: {betat.shape}')
        # print(f'gammat2 shape: {gammat.shape}')
        # print('\n')
        xc = xc*betat + gammat #FiLM style embedding
        xc = self.conv2(xc)
        
        # Add correct residual connection
        out = xc + self.skip_conv(x)
        
        # Normalize by sqrt(2) to maintain variance
        return out / 1.414
    
class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int=None, context_dim: int=256, num_res: int=2):
        super().__init__()
        
        self.conv1 = ResidualConvBlock(in_channels, out_channels)
        self.attn1 = AttentionBlock(out_channels, context_dim, mlp_dim=out_channels*2, heads=num_heads) if num_heads is not None else nn.Identity()

        self.rem_blocks = []
        for _ in range(num_res-1):
            self.rem_blocks.append(ResidualConvBlock(out_channels, out_channels))
            if num_heads is not None:
                self.rem_blocks.append(AttentionBlock(out_channels, context_dim, mlp_dim=out_channels*2, heads=num_heads))
        self.rem_blocks = nn.ModuleList(self.rem_blocks)
        self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)



    def forward(self, x: torch.Tensor, c_attn: torch.Tensor, c_embed: torch.Tensor, tin: torch.Tensor, tout: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor of shape (B, in_channels, H, W)
        c_attn: Context tensor for cross attention of shape (B, context_dim)
        c_embed: Context embedding tensor of shape (B, in_channels*2)
        tin: Time embedding tensor of shape (B, in_channels*2)
        tout: Time embedding tensor of shape (B, out_channels*2)
        Returns:
            x: Output tensor of shape (B, out_channels, H/2, W/2)
        """
 
        x = self.conv1(x, c_embed, tin, tout) # B, in_channels, H, W -> B, out_channels, H, W

        x = self.attn1(x, c_attn) if isinstance(self.attn1, AttentionBlock) else self.attn1(x) # B, out_channels, H, W -> B, out_channels, H, W

        for block in self.rem_blocks:
            if isinstance(block, ResidualConvBlock):
                # Choosing not to use context for residual_block 2+
                x = block(x, None, tout, tout) # B, out_channels, H, W -> B, out_channels, H, W
            else:
                # Need context in original dim for cross attention
                x = block(x, c_attn) # B, out_channels, H, W -> B, out_channels, H, W

        skip = x
        x = self.down_conv(x) # B, out_channels, H, W -> B, out_channels, H/2, W/2
        return x, skip

class UnetUp(nn.Module):
    """
    Upsampling block for U-Net architecture.
    """
    def __init__(self, in_channels: int, out_channels: int, num_heads: int=None, context_dim: int=256, num_res: int=2):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = ResidualConvBlock(2*out_channels, out_channels)
        self.attn1 = AttentionBlock(out_channels, context_dim, mlp_dim=out_channels*2, heads=num_heads) if num_heads is not None else nn.Identity()
        self.rem_blocks = []
        for _ in range(num_res-1):
            self.rem_blocks.append(ResidualConvBlock(out_channels, out_channels))
            if num_heads is not None:
                self.rem_blocks.append(AttentionBlock(out_channels, context_dim, mlp_dim=out_channels*2, heads=num_heads))
        self.rem_blocks = nn.ModuleList(self.rem_blocks)

    def forward(self, x: torch.Tensor, c_attn, c_embed, tin , tout, skip) -> torch.Tensor:
        """
        x: Input tensor of shape (B, in_channels, H, W)
        c_attn: Context tensor for cross attention of shape (B, context_dim)
        c_embed: Context embedding tensor of shape (B, in_channels*2)
        tin: Time embedding tensor of shape (B, (2*out_channels)*2)
        tout: Time embedding tensor of shape (B, out_channels*2)
        skip: Skip tensor of shape (B, out_channels, H, W)
        Returns:
            x: Output tensor of shape (B, out_channels, H*2, W*2)
        """
        x = self.up_conv(x) # B, in_channels, H, W -> B, out_channels, H*2, W*2
        x = torch.cat((x, skip), 1) / 1.414 # B, out_channels, H, W -> B, 2*out_channels, H, W
        x = self.conv1(x, c_embed, tin, tout) # B, 2*out_channels, H, W -> B, out_channels, H, W
        x = self.attn1(x, c_attn) if isinstance(self.attn1, AttentionBlock) else self.attn1(x) # B, out_channels, H, W -> B, out_channels, H, W
        for block in self.rem_blocks:
            if isinstance(block, ResidualConvBlock):
                x = block(x, None, tout, tout) # B, out_channels, H, W -> B, out_channels, H, W
            else:
                x = block(x, c_attn) # B, out_channels, H, W -> B, out_channels, H, W

        return x

class UNetBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, context_dim: int):
        super().__init__()
        self.conv1 = ResidualConvBlock(in_channels, out_channels)
        self.attn1 = AttentionBlock(out_channels, context_dim, mlp_dim=out_channels*2, heads=8)
        self.conv2 = ResidualConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, c_attn, c_embed, t) -> torch.Tensor:
        x = self.conv1(x, c_embed, t, t) # B, in_channels, H, W -> B, out_channels, H, W
        x = self.attn1(x, c_attn) # B, out_channels, H, W -> B, out_channels, H, W
        x = self.conv2(x, c_embed, t, t) # B, out_channels, H, W -> B, out_channels, H, W
        return x
        
class TokenizeContext(nn.Module):
    """
    Turns context vector of shape (B, context_len) into a tensor of shape (B, context_len, context_dim) for cross attention.
    """
    def __init__(self, context_len: int, context_dim: int):
        super().__init__()
        self.context_len = context_len
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.Linear(context_len, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        return self.net(x)
    
class EmbedFC(nn.Module):
    """
    Fully connected embedding network.
    """
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)


# class ContextUnet(nn.Module):
#     """
#     U-Net architecture with context embeddings for conditional diffusion.
#     """
#     def __init__(self, in_feats: int, context_len: int, n_feat: int = 64, time_dim: int=128, context_dim: int=64, channel_mults = (1,2,2,4)):
#         super().__init__()
#         self.in_feats = in_feats
#         self.n_feat = n_feat
#         self.channel_mults = channel_mults
#         # Initial convolution
#         self.init_conv = nn.Conv2d(in_feats, n_feat, kernel_size=3, stride=1, padding=1)
        
#         # Downsampling path
#         self.downs = []
#         in_channels = n_feat*channel_mults[0]
#         for i, mult in enumerate(channel_mults):
#             # Use attention after first 2 downsampling blocks
#             out_channels = n_feat * mult
#             if i < 2: 
#                 self.downs.append(UnetDown(in_channels, out_channels, context_dim=context_dim, num_res=2, num_heads=None))
#             else:
#                 self.downs.append(UnetDown(in_channels, out_channels, context_dim=context_dim, num_res=2, num_heads=8))
#             in_channels = out_channels
#         self.downs = nn.ModuleList(self.downs)
        
#         # Bottleneck
#         in_channels = n_feat*channel_mults[-1]
#         self.bottleneck = UNetBottleneck(in_channels, in_channels, context_dim)

#         # Upsampling path
#         self.ups = []
#         in_channels = n_feat*channel_mults[-1]
#         for i, mult in enumerate(reversed(channel_mults)):
#             out_channels = n_feat * mult
#             if i < len(channel_mults) - 2:
#                 self.ups.append(UnetUp(in_channels, out_channels, context_dim=context_dim, num_res=2, num_heads=8))
#             else:
#                 self.ups.append(UnetUp(in_channels, out_channels, context_dim=context_dim, num_res=2, num_heads=None))
#             in_channels = out_channels
#         self.ups = nn.ModuleList(self.ups)
        
#         # Final convolution
#         self.out = nn.Sequential(nn.GroupNorm(8, in_channels), nn.SiLU(), nn.Conv2d(in_channels, in_feats, kernel_size=3, stride=1, padding=1))
        
        
#         # Time and context embeddings
#         self.time_fourier = FourierEncoder(time_dim)
#         self.attn_context_embed = TokenizeContext(1, context_dim)
#         self.time_embeds = nn.ModuleList([EmbedFC(time_dim, num_feats*2*n_feat) for num_feats in channel_mults])
#         self.time_embeds.append(EmbedFC(time_dim, n_feat*4*channel_mults[-1])) #need extra for concatenated
#         self.context_embeds = nn.ModuleList([EmbedFC(context_len, num_feats*2*n_feat) for num_feats in channel_mults])
#         self.context_embeds.append(EmbedFC(context_len, n_feat*4*channel_mults[-1]))

#     def build_film_mlps(n_feat, channel_mults, time_dim, context_len):
#         """
#         Return two ModuleDicts: time_mlps[C], context_mlps[C] that output 2*C features
#         (FiLM: beta, gamma) for any channel count C seen in the net.
#         """
#         film_channels = set()

#         # downs
#         in_ch = n_feat * channel_mults[0]
#         for mult in channel_mults:
#             out_ch = n_feat * mult
#             film_channels.add(in_ch)   # tin for first conv
#             film_channels.add(out_ch)  # tout for second conv
#             in_ch = out_ch

#         # bottleneck
#         bot_ch = n_feat * channel_mults[-1]
#         film_channels.add(bot_ch)

#         # ups (input to the first ResBlock after concat is 2*out_ch)
#         prev_ch = bot_ch
#         for mult in reversed(channel_mults):
#             out_ch = n_feat * mult
#             film_channels.add(prev_ch)             # tin for first up block
#             film_channels.add(out_ch)              # tout for first up block
#             film_channels.add(out_ch * 2)          # because of concat(skip, x)
#             prev_ch = out_ch

#         # build dicts
#         time_mlps    = nn.ModuleDict()
#         context_mlps = nn.ModuleDict() 
#         for C in sorted(film_channels):
#             key = str(C)
#             time_mlps[key]    = EmbedFC(time_dim,    2 * C)
#             context_mlps[key] = EmbedFC(context_len, 2 * C)

#         return time_mlps, context_mlps

#     def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the network.
        
#         Args:
#             x: Input image (noisy)
#             c: Context conditioning
#             t: Diffusion Timestep
#             context_mask: Mask for classifier-free guidance
            
#         Returns:
#             Predicted noise
#         """
#         # Process input through initial convolution and downsample

#         # print(f'x shape: {x.shape}, c shape: {c.shape}, t shape: {t.shape}, context_mask shape: {context_mask.shape}')
#         # print('\n')
#         x = self.init_conv(x)
#         # print(f'x shape after init_conv: {x.shape}')
#         # print('\n')
#         fourier_t = self.time_fourier(t)
#         # print(f'fourier_t shape: {fourier_t.shape}')
#         # print('\n')
#         # Reshape context for processing
#         c = c.reshape((c.shape[0], c.shape[2]))
#         # Apply context mask for classifier-free guidance
#         context_mask = context_mask.reshape((x.shape[0], c.shape[1]))
#         context_mask = -1 * (1 - context_mask)  # Flip 0 <-> 1
#         c = c * context_mask
#         # print(f'c shape after reshape: {c.shape}')
#         # print('\n')
#         c_attn = self.attn_context_embed(c)
#         # print(f'c_attn shape: {c_attn.shape}')
#         # print('\n')
#         t_embed = [self.time_embeds[i](fourier_t) for i in range(len(self.time_embeds))]
#         c_embed = [self.context_embeds[i](c) for i in range(len(self.context_embeds))]

#         tin = t_embed[0]
#         cin = c_embed[0]
#         skips = []
#         for i, down in enumerate(self.downs):
#             # print(f'x shape before down mult {self.channel_mults[i]}: {x.shape}')
#             # print('\n')
#             # print(f't_embed[i] shape: {t_embed[i].shape}')
#             # print(f'c_embed[i] shape: {cin.shape}')
#             x, skip = down(x, c_attn, cin, tin, t_embed[i])
#             # print(f'x shape after down mult {self.channel_mults[i]}: {x.shape}')
#             tin = t_embed[i]
#             cin = c_embed[i]
#             # print('\n')
#             # print(f'tin shape: {tin.shape}')
#             # print('\n')
#             skips.append(skip)
#         # print(f'x shape before bottleneck: {x.shape}')
#         # print('\n')
#         x = self.bottleneck(x, c_attn=c_attn, c_embed=c_embed[-2], t=t_embed[-2]) #second to last because of concat
#         # print(f'x shape after bottleneck: {x.shape}')
#         # print('\n')
#         tin = t_embed[-1]
#         cin = c_embed[-1]
#         c_embed_in = [c_embed[-1], c_embed[-2], c_embed[-2], c_embed[-3], None]
#         t_embed_in = [t_embed[-1], t_embed[-2], t_embed[-2], t_embed[-3], None]
#         for i, up in enumerate(self.ups):
#             # print(f'x shape before up mult {self.channel_mults[i]}: {x.shape}')
#             # print('\n')
#             # print(f'skips shape: {skips[-(i+1)].shape}')
#             # print(f'c_embed shape: {cin.shape}')
#             # print(f't_embed shape: {t_embed[-(i+1)].shape}')
#             # print(f't_embed[-(i+2)] shape: {t_embed[-(i+2)].shape}')
#             # print(f't shape: {tin.shape}')
#             # print('\n')
#             x = up(x, c_attn=c_attn, c_embed=cin, tin=tin, tout=t_embed[-(i+2)], skip=skips[-(i+1)])
#             tin = t_embed_in[i+1]
#             cin = c_embed_in[i+1]
            
#             # print(f'x shape after up mult {self.channel_mults[i]}: {x.shape}')
#             # print('\n')
#             # print('\n')
#         x = self.out(x)
#         # print(f'x shape after out: {x.shape}')
#         # print('\n')
#         # return None
#         return x


def build_film_mlps(n_feat, channel_mults, time_dim, context_len):
    film_channels = set()

    # downs
    in_ch = n_feat * channel_mults[0]
    for mult in channel_mults:
        out_ch = n_feat * mult
        film_channels.add(in_ch)
        film_channels.add(out_ch)
        in_ch = out_ch

    # bottleneck
    bot_ch = n_feat * channel_mults[-1]
    film_channels.add(bot_ch)

    # ups
    prev_ch = bot_ch
    for mult in reversed(channel_mults):
        out_ch = n_feat * mult
        film_channels.add(prev_ch)
        film_channels.add(out_ch)
        film_channels.add(out_ch*2)   # after concat
        prev_ch = out_ch

    time_mlps    = nn.ModuleDict()
    context_mlps = nn.ModuleDict()
    for C in sorted(film_channels):
        k = str(C)
        time_mlps[k]    = EmbedFC(time_dim,    2 * C)
        context_mlps[k] = EmbedFC(context_len, 2 * C)
    return time_mlps, context_mlps

class ContextUnet(nn.Module):
    def __init__(self, in_feats, context_len,
                 n_feat=64, time_dim=128, context_dim=64,
                 channel_mults=(1,2,2,4), heads_at=(2,3), num_res=(2,2,2,2)):
        super().__init__()
        self.in_feats = in_feats
        self.n_feat = n_feat
        self.channel_mults = channel_mults
        self.num_res = num_res
        self.init_conv = nn.Conv2d(in_feats, n_feat, 3, 1, 1)

        print(f'num_res: {num_res}')
        print(f'channel_mults: {channel_mults}')
        print(f'heads_at: {heads_at}')
        print(f'n_feat: {n_feat}')
        print(f'time_dim: {time_dim}')
        print(f'context_dim: {context_dim}')
        print(f'context_len: {context_len}')
        print(f'in_feats: {in_feats}')
        # Embeddings
        self.time_fourier = FourierEncoder(time_dim)
        self.attn_context_embed = TokenizeContext(1, context_dim)
        self.time_mlps, self.context_mlps = build_film_mlps(
            n_feat, channel_mults, time_dim, context_len
        )

        # Downs
        self.downs = nn.ModuleList()
        in_ch = n_feat * channel_mults[0]
        for i, mult in enumerate(channel_mults):
            out_ch = n_feat * mult
            num_heads = 8 if i in heads_at else None
            self.downs.append(UnetDown(in_ch, out_ch, num_heads=num_heads,
                                       context_dim=context_dim, num_res=self.num_res[i]))
            in_ch = out_ch

        # Bottleneck
        bot_ch = n_feat * channel_mults[-1]
        self.bottleneck = UNetBottleneck(bot_ch, bot_ch, context_dim)

        # Ups
        self.ups = nn.ModuleList()
        prev_ch = bot_ch
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = n_feat * mult
            stage_idx = len(channel_mults) - 1 - i
            num_heads = 8 if stage_idx in heads_at else None
            self.ups.append(UnetUp(prev_ch, out_ch, num_heads=num_heads,
                                    context_dim=context_dim, num_res=self.num_res[i]))
            prev_ch = out_ch

        self.out = nn.Sequential(
            nn.GroupNorm(8, prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, in_feats, 3, 1, 1)
        )

    def film(self, mlp_dict: nn.ModuleDict, C: int, x: torch.Tensor) -> torch.Tensor:
        """
        Helper to get embeddings for corresponding channel count.
        """
        return mlp_dict[str(C)](x)

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)

        fourier_t = self.time_fourier(t)

        c = c.reshape(c.shape[0], c.shape[2])
        context_mask = context_mask.reshape(x.shape[0], c.shape[1])
        c = c * (-1 * (1 - context_mask))
        c_attn = self.attn_context_embed(c)

        skips = []

        # Downs
        in_ch = self.n_feat * self.channel_mults[0]
        for i, down in enumerate(self.downs):
            out_ch = self.n_feat * self.channel_mults[i]

            tin  = self.film(self.time_mlps, in_ch,  fourier_t)
            tout = self.film(self.time_mlps, out_ch, fourier_t)
            c_embed  = self.film(self.context_mlps, in_ch,  c)  

            x, skip = down(x, c_attn=c_attn, c_embed=c_embed, tin=tin, tout=tout)
            skips.append(skip)
            in_ch = out_ch

        # Bottleneck
        bot_ch = self.n_feat * self.channel_mults[-1]
        t_bot  = self.film(self.time_mlps,    bot_ch, fourier_t)
        c_bot  = self.film(self.context_mlps, bot_ch, c)
        x = self.bottleneck(x, c_attn=c_attn, c_embed=c_bot, t=t_bot)

        # Ups
        for i, up in enumerate(self.ups):
            out_ch = self.n_feat * self.channel_mults[-(i+1)]
            in_cat_ch = out_ch*2   # because of concat with skip

            tin  = self.film(self.time_mlps,    in_cat_ch, fourier_t)
            tout = self.film(self.time_mlps,    out_ch,    fourier_t)
            c_embed  = self.film(self.context_mlps, in_cat_ch, c)

            x = up(x, c_attn=c_attn, c_embed=c_embed, tin=tin, tout=tout, skip=skips.pop())

        return self.out(x)
