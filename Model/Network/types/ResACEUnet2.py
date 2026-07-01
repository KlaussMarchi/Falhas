"""
ResACE_Unet — Residual Attention-augmented Channel-Enhanced 3D U-Net
=====================================================================

Optimized for binary fault segmentation in 3D seismic volumes (128³).

Changes from the original implementation and rationale:
-------------------------------------------------------
1. BUGFIX — Transformer self-attention scaling: was dividing by C**0.5 instead of
   (C//num_heads)**0.5, causing over-sharpened attention weights and poor gradient flow.
2. GroupNorm replaces BatchNorm3d — stabilizes training with batch_size=2 (the project's
   standard setting) where BatchNorm statistics are extremely noisy.
3. LeakyReLU replaces ReLU — prevents dead neurons in residual paths; matches the
   better-performing Unet3D_V2 architecture in this repository.
4. Kaiming (Conv3d) + trunc_normal (Linear/attention) weight initialization — critical
   for the hybrid CNN-Transformer architecture to converge reliably.
5. DropPath (Stochastic Depth) regularization — proven technique for vision transformers
   with limited training data, complementing the existing Dropout3d.
6. Multi-head self-attention (4 heads) replaces single-head — richer attention patterns
   for capturing diverse fault orientations and geometries.
7. Deep supervision — optional intermediate decoder outputs that improve gradient flow
   to early layers and help delineate thin fault boundaries.
8. Graduated dropout — lower rates in shallow encoder (preserving texture features),
   higher in deep encoder (regularizing abstract features).
9. 4 encoder levels (was 3) — expands receptive field to match Unet3D_V2 depth; bottleneck
   at 8³ spatial with 512 tokens is efficient for transformer attention.
10. Backward-compatible API — identical __init__ signature as before:
    ResACE_Unet(in_channels, num_classes, base_filters, dropout_rate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# ─────────────────────────────────────────────────────────────────────────────
# Utility: DropPath (Stochastic Depth)
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample during training.
    
    Randomly drops entire residual branches with probability `drop_prob`,
    improving regularization for limited-data scenarios.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape: (B, 1, 1, 1, 1) for broadcasting over (B, C, D, H, W)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helper
# ─────────────────────────────────────────────────────────────────────────────

def get_norm_3d(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with automatic group count adjustment.
    
    GroupNorm is preferred over BatchNorm3d because this project trains with
    batch_size=2, making BatchNorm statistics extremely noisy.
    """
    # Ensure num_groups divides num_channels evenly
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


# ─────────────────────────────────────────────────────────────────────────────
# Channel Attention (Squeeze-and-Excitation, 3D)
# ─────────────────────────────────────────────────────────────────────────────

class EfficientChannelAttention3D(nn.Module):
    """Squeeze-and-Excitation channel attention for 3D feature maps.
    
    Adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels.
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1, 1)
        return x * y


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention for 3D (Bottleneck)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention3D(nn.Module):
    """Multi-head self-attention for 3D feature maps.
    
    Used at the bottleneck where the spatial volume is small (e.g. 8³ = 512 tokens).
    Captures long-range dependencies across the entire volume, essential for
    detecting faults that span large spatial extents.
    
    Key fix from original: scaling factor uses head_dim (C // num_heads), not C.
    """
    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        assert in_channels % num_heads == 0, \
            f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        # Correct scaling: 1/sqrt(head_dim), NOT 1/sqrt(in_channels)
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv3d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W  # number of spatial tokens
        
        # Compute Q, K, V in one pass
        qkv = self.qkv(x)                                   # [B, 3C, D, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)  # [B, 3, heads, head_dim, N]
        q, k, v = qkv.unbind(dim=1)                          # each: [B, heads, head_dim, N]
        
        q = q.transpose(-2, -1)  # [B, heads, N, head_dim]
        k = k                    # [B, heads, head_dim, N]
        v = v.transpose(-2, -1)  # [B, heads, N, head_dim]
        
        # Scaled dot-product attention (correct scaling by head_dim)
        attn = (q @ k) * self.scale                          # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v)                                     # [B, heads, N, head_dim]
        out = out.transpose(-2, -1)                          # [B, heads, head_dim, N]
        out = out.reshape(B, C, D, H, W)
        out = self.proj(out)
        
        # Learnable residual gating
        return x + self.gamma * out


# ─────────────────────────────────────────────────────────────────────────────
# ResACE Block (Residual + Attention + Conv)
# ─────────────────────────────────────────────────────────────────────────────

class ResACEBlock3D(nn.Module):
    """Hybrid block combining residual convolution with attention.
    
    Architecture per block:
        input → Conv3d → GroupNorm → LeakyReLU → Dropout3d →
                Conv3d → GroupNorm → [Attention] → + residual → LeakyReLU → [DropPath]
    
    Attention type:
        - Channel (SE): default for encoder/decoder — efficient, local
        - Multi-Head Self-Attention: for bottleneck — global context
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dropout_rate: float = 0.0, use_transformer: bool = False,
                 num_heads: int = 4, drop_path_rate: float = 0.0):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = get_norm_3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = get_norm_3d(out_channels)
        
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        # Residual projection when dimensions change
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm_3d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        # Attention mechanism
        if use_transformer:
            self.attention = MultiHeadSelfAttention3D(out_channels, num_heads=num_heads)
        else:
            self.attention = EfficientChannelAttention3D(out_channels)
        
        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Apply attention
        out = self.attention(out)
        
        # Residual connection with stochastic depth
        out = residual + self.drop_path(out)
        out = self.act(out)
        
        return out


# ─────────────────────────────────────────────────────────────────────────────
# ResACE U-Net (Main Model)
# ─────────────────────────────────────────────────────────────────────────────

class ResACE_Unet(nn.Module):
    """ResACE U-Net: Hybrid 3D U-Net with Residual Attention and Channel Enhancement.
    
    A 4-level encoder-decoder architecture with:
    - SE channel attention in encoder/decoder blocks
    - Multi-head self-attention at the bottleneck
    - GroupNorm for batch_size=2 stability
    - DropPath (stochastic depth) regularization
    - Graduated dropout rates
    - Deep supervision option for improved gradient flow
    
    Args:
        in_channels: Number of input channels (1 for seismic amplitude).
        num_classes: Number of output classes (1 for binary fault detection).
        base_filters: Base filter count (doubled at each level).
        dropout_rate: Base dropout rate (graduated across levels).
        deep_supervision: If True, returns list of outputs at multiple scales.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1,
                 base_filters: int = 16, dropout_rate: float = 0.2,
                 deep_supervision: bool = False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        f = base_filters
        filters = [f, f * 2, f * 4, f * 8, f * 16]
        
        # Graduated dropout: lower early (preserve features), higher deep (regularize)
        dr = dropout_rate
        enc_dropouts = [dr * 0.25, dr * 0.5, dr * 1.0, dr * 1.5]
        
        # Stochastic depth: conservative linearly increasing drop rate
        dp_rates = [0.0, 0.02, 0.05, 0.1]
        
        # ── ENCODER (4 levels) ─────────────────────────────────────────────────
        self.enc1 = ResACEBlock3D(in_channels, filters[0], enc_dropouts[0], drop_path_rate=dp_rates[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ResACEBlock3D(filters[0], filters[1], enc_dropouts[1], drop_path_rate=dp_rates[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ResACEBlock3D(filters[1], filters[2], enc_dropouts[2], drop_path_rate=dp_rates[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc4 = ResACEBlock3D(filters[2], filters[3], enc_dropouts[3], drop_path_rate=dp_rates[3])
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # ── BOTTLENECK (Transformer attention for global context) ──────────────
        # At 128³ input with 4 pools: spatial = 8³ = 512 tokens — efficient for attention
        num_heads = max(4, filters[4] // 64)  # ensure enough heads for the channel count
        # Ensure divisibility
        while filters[4] % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.bottleneck = ResACEBlock3D(
            filters[3], filters[4], dr * 2.0,
            use_transformer=True, num_heads=num_heads,
            drop_path_rate=0.15
        )
        
        # ── DECODER (4 levels) ─────────────────────────────────────────────────
        self.upconv4 = nn.ConvTranspose3d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4 = ResACEBlock3D(filters[4], filters[3], dr, drop_path_rate=dp_rates[3])
        
        self.upconv3 = nn.ConvTranspose3d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = ResACEBlock3D(filters[3], filters[2], dr, drop_path_rate=dp_rates[2])
        
        self.upconv2 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = ResACEBlock3D(filters[2], filters[1], dr * 0.5, drop_path_rate=dp_rates[1])
        
        self.upconv1 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = ResACEBlock3D(filters[1], filters[0], dr * 0.25, drop_path_rate=dp_rates[0])
        
        # ── OUTPUT ─────────────────────────────────────────────────────────────
        self.out_conv = nn.Conv3d(filters[0], num_classes, kernel_size=1)
        
        # Deep supervision heads (optional)
        if deep_supervision:
            self.ds_conv4 = nn.Conv3d(filters[3], num_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv3d(filters[2], num_classes, kernel_size=1)
            self.ds_conv2 = nn.Conv3d(filters[1], num_classes, kernel_size=1)
        
        # ── INITIALIZE WEIGHTS ─────────────────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights following best practices for hybrid CNN-Transformer.
        
        - Conv3d: Kaiming normal (fan_out, LeakyReLU mode)
        - Linear: truncated normal (std=0.02) — standard for attention layers
        - GroupNorm: weight=1, bias=0
        """
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Pad to multiple of 16 (4 pools of stride 2 → divisible by 2⁴=16) ──
        d, h, w = x.shape[2:]
        pad_d = (16 - d % 16) % 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        
        # ── Encoder ────────────────────────────────────────────────────────────
        e1 = self.enc1(x)       # [B, f, D, H, W]
        p1 = self.pool1(e1)     # [B, f, D/2, H/2, W/2]
        
        e2 = self.enc2(p1)      # [B, 2f, D/2, H/2, W/2]
        p2 = self.pool2(e2)     # [B, 2f, D/4, H/4, W/4]
        
        e3 = self.enc3(p2)      # [B, 4f, D/4, H/4, W/4]
        p3 = self.pool3(e3)     # [B, 4f, D/8, H/8, W/8]
        
        e4 = self.enc4(p3)      # [B, 8f, D/8, H/8, W/8]
        p4 = self.pool4(e4)     # [B, 8f, D/16, H/16, W/16]
        
        # ── Bottleneck (with multi-head self-attention) ────────────────────────
        b = self.bottleneck(p4)  # [B, 16f, D/16, H/16, W/16]
        
        # ── Decoder with skip connections ──────────────────────────────────────
        u4 = self.upconv4(b)
        u4 = torch.cat([u4, e4], dim=1)  # [B, 16f, D/8, H/8, W/8]
        d4 = self.dec4(u4)                # [B, 8f, D/8, H/8, W/8]
        
        u3 = self.upconv3(d4)
        u3 = torch.cat([u3, e3], dim=1)  # [B, 8f, D/4, H/4, W/4]
        d3 = self.dec3(u3)                # [B, 4f, D/4, H/4, W/4]
        
        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, e2], dim=1)  # [B, 4f, D/2, H/2, W/2]
        d2 = self.dec2(u2)                # [B, 2f, D/2, H/2, W/2]
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)  # [B, 2f, D, H, W]
        d1 = self.dec1(u1)                # [B, f, D, H, W]
        
        # ── Output ─────────────────────────────────────────────────────────────
        out = self.out_conv(d1)
        
        # Crop padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            out = out[:, :, :d, :h, :w]
        
        # Deep supervision: return multi-scale outputs during training
        if self.deep_supervision and self.training:
            ds4 = F.interpolate(self.ds_conv4(d4), size=(d, h, w), mode='trilinear', align_corners=False)
            ds3 = F.interpolate(self.ds_conv3(d3), size=(d, h, w), mode='trilinear', align_corners=False)
            ds2 = F.interpolate(self.ds_conv2(d2), size=(d, h, w), mode='trilinear', align_corners=False)
            return [out, ds4, ds3, ds2]
        
        return out