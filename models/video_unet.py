from typing import Iterable, Literal, Tuple
import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as FF
from . import encoding, ops
from mamba_ssm import Mamba
import math
import matplotlib.pyplot as plt

def _join(*tensors) -> torch.Tensor:
    return torch.cat(tensors, dim=1)

def _n_tuple(x: Iterable | int, N: int) -> tuple[int, ...]:
    if isinstance(x, Iterable):
        assert len(x) == N
        return tuple(x)
    else:
        return (x,) * N

def compute_frame_diff(x):
    """Compute frame differences."""
    B, C, F, H, W = x.shape
    if F <= 1:
        return torch.zeros_like(x)
    
    diff = torch.zeros_like(x)
    diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    diff[:, :, 0] = diff[:, :, 1]
    return diff

class StaticBranch(nn.Module):
    """Static branch: long-term temporal averaging."""
    def __init__(self, in_channels, feature_dim=64, num_frames=12, avg_window=5):
        super().__init__()
        self.avg_window = avg_window
        
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, feature_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, F, H, W = x.shape
        
        averaged_x = torch.zeros_like(x)
        for i in range(F):
            start_idx = max(0, i - self.avg_window // 2)
            end_idx = min(F, i + self.avg_window // 2 + 1)
            averaged_x[:, :, i] = x[:, :, start_idx:end_idx].mean(dim=2)
        
        static_feat = self.feature_extractor(averaged_x)
        return static_feat

class DynamicBranch(nn.Module):
    """Dynamic branch: leverages frame differences."""
    def __init__(self, in_channels, feature_dim=64):
        super().__init__()
        
        input_channels = in_channels * 2  # original + frame diff
        
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, feature_dim, kernel_size=(3, 3, 3), padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        frame_diff = compute_frame_diff(x)
        dynamic_input = torch.cat([x, frame_diff], dim=1)
        dynamic_feat = self.feature_extractor(dynamic_input)
        return dynamic_feat

class SimpleDualBranch(nn.Module):
    """Simplified dual-branch network with dynamic output channel support."""
    def __init__(self, in_channels, output_channels, feature_dim=64, num_frames=12):
        super().__init__()
        
        self.static_branch = StaticBranch(in_channels, feature_dim, num_frames)
        self.dynamic_branch = DynamicBranch(in_channels, feature_dim)
        
        # Fusion output channels should match the target output
        self.fusion_conv = nn.Conv3d(feature_dim * 2, output_channels, kernel_size=1)
        
    def forward(self, x):
        static_feat = self.static_branch(x)
        dynamic_feat = self.dynamic_branch(x)
        fused_feat = self.fusion_conv(torch.cat([static_feat, dynamic_feat], dim=1))
        return dynamic_feat, static_feat, fused_feat

def deformable_sampling_3d(x_seq, sampling_grid, F, H, W):
    """
    Args:
        x_seq: (B, F*H*W, C) -> (B, C, F, H, W)
        sampling_grid: (B, F*H*W, 3) in [-1, 1] range
    Returns:
        sampled_feat: (B, F*H*W, C)
    """
    B, L, C = x_seq.shape
    x_grid = x_seq.view(B, F, H, W, C).permute(0, 4, 1, 2, 3)  # (B, C, F, H, W)
    sampling_grid = sampling_grid.view(B, F, H, W, 3)  # (B, F, H, W, 3)

    # grid: (B, F, H, W, 3) -> [-1, 1]
    sampled_feat = FF.grid_sample(
        x_grid, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True
    )  # (B, C, F, H, W)
    sampled_feat = sampled_feat.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)  # (B, F*H*W, C)
    return sampled_feat

def deformable_sampling_2d(x_seq, sampling_grid, F, H, W):
    """
    Args:
        x_seq: (B, F*H*W, C) -> (B, C, F, H, W)
        sampling_grid: (B, F*H*W, 2) in [-1, 1] range
    Returns:
        sampled_feat: (B, F*H*W, C)
    """
    B, L, C = x_seq.shape
    x_grid = x_seq.view(B, F, H, W, C).permute(0, 4, 1, 2, 3)  # (B, C, F, H, W)
    
    # Perform 2D sampling per temporal frame
    sampled_feats = []
    for f in range(F):
        frame_feat = x_grid[:, :, f, :, :]  # (B, C, H, W)
        frame_grid = sampling_grid[:, f*H*W:(f+1)*H*W, :].view(B, H, W, 2)
        # 2D grid_sample requires grid shape (B, H, W, 2)
        sampled_frame = FF.grid_sample(
            frame_feat, frame_grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # (B, C, H, W)
        sampled_feats.append(sampled_frame)
    
    sampled_grid = torch.stack(sampled_feats, dim=2)
    sampled_feat = sampled_grid.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)
    return sampled_feat

class DeformableMambaBranch(nn.Module):
    """Single deformable Mamba branch adapted for 3D sequences."""
    def __init__(self, dim, d_state=64, feature_dim=64, is_dynamic=True):
        super().__init__()
        self.dim = dim
        self.feature_dim = feature_dim
        self.is_dynamic = is_dynamic

        conv_size = 3
        
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=conv_size,
            expand=2,
            dt_rank=math.ceil(dim / 16),
            dt_min=0.001,
            dt_max=0.1,
        )

        if is_dynamic:
            # Dynamic branch: predict 3D offsets
            self.offset_predictor = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
                nn.Tanh()
            )
        else:
            # Static branch: predict 2D offsets
            self.offset_predictor = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
                nn.Tanh()
            )

        self.modulator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, dim),
            nn.Sigmoid()
        )

        pos_dim = 3 if is_dynamic else 2
        self.pos_enc = nn.Linear(pos_dim, dim)

    def _get_base_positions(self, B: int, F: int, H: int, W: int, device: torch.device):
        """Generate base position coordinates in [0, 1]."""
        f_idx = torch.linspace(0, 1, F, device=device)
        h_idx = torch.linspace(0, 1, H, device=device)
        w_idx = torch.linspace(0, 1, W, device=device)

        if self.is_dynamic:
            f_grid, h_grid, w_grid = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
            pos = torch.stack([f_grid, h_grid, w_grid], dim=-1).flatten(0, 2)
            return pos.unsqueeze(0).repeat(B, 1, 1)  # (B, F*H*W, 3)
        else:
            h_grid2d, w_grid2d = torch.meshgrid(h_idx, w_idx, indexing="ij")
            pos = torch.stack([h_grid2d, w_grid2d], dim=-1).flatten(0, 1)
            pos = pos.unsqueeze(0).unsqueeze(0).repeat(B, F, 1, 1)
            return pos.view(B, F * H * W, 2)

    def _deformable_sampling(self, x_seq: torch.Tensor, sampling_pos: torch.Tensor, F: int, H: int, W: int):
        """Perform the actual input feature sampling."""
        if self.is_dynamic:
            # sampling_pos: (B, F*H*W, 3) in [0, 1] -> [-1, 1]
            sampling_pos = sampling_pos * 2 - 1
            return deformable_sampling_3d(x_seq, sampling_pos, F, H, W)
        else:
            # 2D deformable sampling (spatial dimensions only)
            # sampling_pos: (B, F*H*W, 2)
            sampling_pos = sampling_pos * 2 - 1
            return deformable_sampling_2d(x_seq, sampling_pos, F, H, W)

    def forward(self, x_seq: torch.Tensor, branch_feat: torch.Tensor, F: int, H: int, W: int):
        """
        Args:
            x_seq: (B, F*H*W, dim) main feature sequence
            branch_feat: (B, feature_dim, F, H, W) branch features
            F/H/W: spatiotemporal dimensions
        """
        B, seq_len, dim = x_seq.shape
        device = x_seq.device

        # Flatten branch features: (B, feature_dim, F, H, W) -> (B, F*H*W, feature_dim)
        branch_flat = einops.rearrange(branch_feat, "B C F H W -> B (F H W) C")

        offset = self.offset_predictor(branch_flat)  # (B, F*H*W, 3 or 2)

        base_pos = self._get_base_positions(B, F, H, W, device)
        sampling_pos = base_pos + offset * 0.1
        sampling_pos = torch.clamp(sampling_pos, 0, 1)

        sampled_x = self._deformable_sampling(x_seq, sampling_pos, F, H, W)

        global_feat = einops.reduce(branch_flat, "B seq C -> B C", "mean")
        mod_factor = self.modulator(global_feat).unsqueeze(1)  # (B, 1, dim)
        modulated_x = sampled_x * mod_factor

        pos_encoding = self.pos_enc(sampling_pos)  # (B, seq_len, dim)

        try:
            mamba_out = self.mamba(modulated_x + pos_encoding)
        except RuntimeError as e:
            if "causal_conv1d" in str(e):
                print("Mamba conv error, using fallback")
                mamba_out = self.mamba(modulated_x) + pos_encoding
            else:
                raise e

        return mamba_out, global_feat

# ==================== Dual-branch Deformable Mamba ====================

class DualDeformableMamba(nn.Module):
    """Dual-branch deformable Mamba module."""
    def __init__(self, dim, d_state=64, feature_dim=64):
        super().__init__()
        self.dim = dim
        self.feature_dim = feature_dim

        # Two independent deformable Mamba branches
        self.dynamic_branch = DeformableMambaBranch(dim, d_state, feature_dim, is_dynamic=True)
        self.static_branch = DeformableMambaBranch(dim, d_state, feature_dim, is_dynamic=False)

        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, dim),
            nn.Sigmoid()
        )

    def forward(self, x_seq: torch.Tensor, dynamic_feat: torch.Tensor, static_feat: torch.Tensor, F: int, H: int, W: int):
        """
        Args:
            x_seq: (B, seq_len, dim) main feature sequence
            dynamic_feat: (B, feature_dim, F, H, W) dynamic branch features
            static_feat: (B, feature_dim, F, H, W) static branch features
            F/H/W: spatiotemporal dimensions
        """
        dynamic_out, dynamic_global = self.dynamic_branch(x_seq, dynamic_feat, F, H, W)

        static_out, static_global = self.static_branch(x_seq, static_feat, F, H, W)

        fusion_input = torch.cat([dynamic_global, static_global], dim=1)
        alpha = self.fusion_gate(fusion_input)  # (B, dim)
        alpha = alpha.unsqueeze(1)  # (B, 1, dim)

        fused_output = alpha * dynamic_out + (1 - alpha) * static_out
        
        return fused_output

# ==================== Enhanced SSM Block ====================

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model 
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
        self.silu = nn.SiLU()
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, F, H, W = x.shape
        assert C == self.input_dim
        x = einops.rearrange(x, "B C F H W -> B C (F H W)")
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        out = einops.rearrange(out, "B C (F H W) -> B C F H W", F=F, H=H, W=W)
        out = self.silu(out)
        return out

class EnhancedSSMBlock(nn.Module):
    def __init__(self, dim, d_state=64, feature_dim=64):
        super().__init__()
        self.dual_deformable_mamba = DualDeformableMamba(dim, d_state, feature_dim)
        self.normal_mamba = PVMLayer(input_dim=dim, output_dim=dim)
        # self.beta = nn.Parameter(torch.tensor(0.0))
        self.fusion_gate = nn.Sequential(
            nn.Conv3d(dim, 1, kernel_size=1),
            nn.Sigmoid()
            )
        
    def forward(self, x, dynamic_feat, static_feat):
        """
        Inputs:
        - x: (B, C, F, H, W) main features
        - dynamic_feat: (B, C_feat, F, H, W) dynamic features
        - static_feat: (B, C_feat, F, H, W) static features
        """
        B, C, F, H, W = x.shape
        
        x_seq = einops.rearrange(x, "B C F H W -> B (F H W) C")
        
        output_dual = self.dual_deformable_mamba(x_seq, dynamic_feat, static_feat, F, H, W)
        
        output_dual = einops.rearrange(output_dual, "B (F H W) C -> B C F H W", F=F, H=H, W=W)

        output_normal = self.normal_mamba(x)

        gate = self.fusion_gate(output_dual)
        output = gate * output_normal + (1 - gate) * output_dual

        return output

class EnhancedBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_residual_blocks,
        emb_channels,
        down: int = 1,
        up: int = 1,
        attn=False,
        ssm=False,
        dual_branch=False,
        num_heads=8,
        gn_num_groups=8,
        gn_eps=1e-6,
        dropout=0.0,
        num_frames=12
    ):
        super().__init__()
        
        self.dual_branch = None
        if dual_branch:
            # Pass current layer's input channels so dual_branch knows how many channels to output
            self.dual_branch = SimpleDualBranch(
                in_channels=in_channels,
                output_channels=in_channels,
                feature_dim=64,
                num_frames=num_frames
            )
            # Feature fusion gate; fused_feat channels match in_channels
            self.fusion_gate = nn.Sequential(
                nn.Conv3d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )

        self.res_blocks = ops.ConditionalSequential()
        for i in range(num_residual_blocks):
            block_in = out_channels if i != 0 or down > 1 else in_channels
            self.res_blocks.append(
                ResidualBlock3D(
                    block_in, out_channels, emb_channels,
                    gn_num_groups, gn_eps, dropout=dropout
                )
            )
        
        self.temp_attn = TemporalAttentionBlock(out_channels, num_heads) if attn else None

        self.ssm = EnhancedSSMBlock(out_channels) if ssm else None

        self.downsample = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                SpatialResample(down=down)
            ) if down > 1 else nn.Identity()
        )

        self.upsample = (
            nn.Sequential(
                SpatialResample(up=up),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
            )
            if up > 1
            else nn.Identity()
        )

    def forward(self, x, temb=None, branch_features=None):
        """
        branch_features: (dynamic_feat, static_feat) features passed from the previous layer
        """
        if self.dual_branch:
            dynamic_feat, static_feat, fused_feat = self.dual_branch(x)
            
            # Adaptive fusion — x and fused_feat channels should match
            gate = self.fusion_gate(fused_feat)
            x = gate * x + (1 - gate) * fused_feat

            # Save branch features for subsequent layers
            branch_features = (dynamic_feat, static_feat)

        if not isinstance(self.downsample, nn.Identity):
            x = self.downsample(x)
        
        x = self.res_blocks(x, temb)

        if self.temp_attn:
            x = x + self.temp_attn(x)
        
        # Enhanced SSM (uses dual-branch features)
        if self.ssm and branch_features is not None:
            dynamic_feat, static_feat = branch_features
            if dynamic_feat.shape[-2:] != x.shape[-2:]:
                dynamic_feat = FF.interpolate(
                    dynamic_feat.view(-1, *dynamic_feat.shape[2:]), 
                    size=x.shape[-2:], mode='bilinear', align_corners=False
                ).view(dynamic_feat.shape[:2] + (-1,) + x.shape[-2:])
                static_feat = FF.interpolate(
                    static_feat.view(-1, *static_feat.shape[2:]), 
                    size=x.shape[-2:], mode='bilinear', align_corners=False
                ).view(static_feat.shape[:2] + (-1,) + x.shape[-2:])
            
            x = self.ssm(x, dynamic_feat, static_feat)
        
        if not isinstance(self.upsample, nn.Identity):
            x = self.upsample(x)

        return x, branch_features


class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.temp_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
    def forward(self, x):
        B, C, F, H, W = x.shape
        x = einops.rearrange(x, "B C F H W -> B (H W F) C")
        x = self.norm(x)
        attn_out, _ = self.temp_attn(x, x, x)
        return einops.rearrange(attn_out, "B (H W F) C -> B C F H W", H=H, W=W)

class SelfAttentionBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        gn_eps: float = 1e-6,
        gn_num_groups: int = 8,
        scale: float = 1 / np.sqrt(2),
    ):
        super().__init__()
        self.norm = nn.GroupNorm(gn_num_groups, in_channels, gn_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn.out_proj.apply(ops.zero_out)
        self.register_buffer("scale", torch.tensor(scale).float())

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, C, F, H, W = h.shape
        h = einops.rearrange(h, "B C F H W -> B (F H W) C")
        h, _ = self.attn(query=h, key=h, value=h, need_weights=False)
        h = einops.rearrange(h, "B (F H W) C -> B C F H W", F=F, H=H, W=W)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.residual(x)
        h = h * self.scale
        return h

class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int | None,
        gn_num_groups: int = 8,
        gn_eps: float = 1e-6,
        scale: float = 1 / np.sqrt(2),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.has_emb = emb_channels is not None

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        if self.has_emb:
            self.norm2 = ops.AdaGN(emb_channels, out_channels, gn_num_groups, gn_eps)
        else:
            self.norm2 = nn.GroupNorm(gn_num_groups, out_channels, gn_eps)
            
        self.silu2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2.apply(ops.zero_out)

        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
        self.norm1 = nn.GroupNorm(gn_num_groups, in_channels, gn_eps)
        self.silu1 = nn.SiLU()

        self.register_buffer("scale", torch.tensor(scale).float())

    def residual(
        self, x: torch.Tensor, emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.silu1(h)
        h = self.conv1(h)
        
        # Handle per-frame emb
        if self.has_emb and emb is not None:
            B, C, F, H, W = h.shape
            if emb.ndim == 3:  # (B, F, dim)
                # Reshape emb to (B, dim, F, 1, 1)
                emb = emb.permute(0, 2, 1)[..., None, None]  # (B, dim, F, 1, 1)
                emb = emb.expand(-1, -1, -1, H, W)  # (B, dim, F, H, W)
            elif emb.ndim == 2:  # (B, dim)
                emb = emb[..., None, None, None]  # (B, dim, 1, 1, 1)
            
            h = self.norm2(h, emb)
        else:
            h = self.norm2(h)
            
        h = self.silu2(h)
        h = self.drop2(h)
        h = self.conv2(h)
        return h

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.skip(x) + self.residual(x, emb)
        h = h * self.scale
        return h

class SpatialResample(nn.Module):
    def __init__(self, down: int = 1, up: int = 1):
        super().__init__()
        self.down = down
        self.up = up
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down == 1 and self.up == 1:
            return x
        
        B, C, F, H, W = x.shape
        x = einops.rearrange(x, 'b c f h w -> (b f) c h w')
        
        if self.down > 1:
            x = nn.functional.avg_pool2d(x, kernel_size=self.down, stride=self.down)
        if self.up > 1:
            x = nn.functional.interpolate(x, scale_factor=self.up, mode='bilinear', align_corners=False)
            
        x = einops.rearrange(x, '(b f) c h w -> b c f h w', f=F)
        return x

def compute_ego_pose_differences(egopose):
    B, _, _ = egopose.shape
    pos_diff = egopose[:, 1:, :] - egopose[:, :-1, :]
    zero_padding = torch.zeros(B, 1, 3, device=egopose.device)
    pos_diff_with_padding = torch.cat([zero_padding, pos_diff], dim=1)
    return pos_diff_with_padding

class EfficientUNet(nn.Module):
    """
    3D U-Net based on a simplified dual-branch structure and deformable Mamba.
    """
    def __init__(
        self,
        in_channels: int,
        resolution: tuple[int, int] | int,
        out_channels: int | None = None,
        base_channels: int = 128,
        temb_channels: int = None,
        channel_multiplier: tuple[int] | int = (1, 2, 4, 8),
        num_residual_blocks: tuple[int] | int = (3, 3, 3, 3),
        gn_num_groups: int = 32 // 4,
        gn_eps: float = 1e-6,
        attn_num_heads: int = 8,
        ego_pose_dim: int = 3,
        coords_encoding: Literal[
            "spherical_harmonics", "polar_coordinates", "fourier_features", None
        ] = "spherical_harmonics",
    ):
        super().__init__()
        self.resolution = _n_tuple(resolution, 2)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        temb_channels = base_channels * 4 if temb_channels is None else temb_channels

        current_in_channels = in_channels
        
        # Spatial coordinate encoding
        coords = encoding.generate_polar_coords(*self.resolution)
        self.register_buffer("coords", coords)
        self.coords_encoding = None
        if coords_encoding == "spherical_harmonics":
            self.coords_encoding = encoding.SphericalHarmonics(levels=5)
            current_in_channels += self.coords_encoding.extra_ch
        elif coords_encoding == "polar_coordinates":
            self.coords_encoding = nn.Identity()
            current_in_channels += coords.shape[1]
        elif coords_encoding == "fourier_features":
            self.coords_encoding = encoding.FourierFeatures(self.resolution)
            current_in_channels += self.coords_encoding.extra_ch

        # Timestep embedding
        self.time_embedding = nn.Sequential(
            ops.SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )

        # Ego pose embedding
        self.ego_pose_dim = ego_pose_dim  # originally 7, now 16 (4x4 matrix flattened)
        self.pose_embedder = nn.Sequential(
            nn.Linear(ego_pose_dim, temb_channels // 2),  # flattened dimension is 16
            nn.SiLU(),
            nn.Linear(temb_channels // 2, temb_channels),
            nn.LayerNorm(temb_channels),
            nn.Tanh()
        )

        self.fuse_proj = nn.Linear(temb_channels * 2, temb_channels)

        updown_levels = 4
        channel_multiplier = _n_tuple(channel_multiplier, updown_levels)
        C = [base_channels] + [base_channels * m for m in channel_multiplier]
        N = _n_tuple(num_residual_blocks, updown_levels)

        cfgs = dict(
            emb_channels=temb_channels,
            gn_num_groups=gn_num_groups,
            gn_eps=gn_eps,
            dropout=0.0,
        )

        self.in_conv = nn.Conv3d(current_in_channels, C[0], 3, 1, 1)

        self.d_block1 = EnhancedBlock3D(C[0], C[1], N[0], **cfgs, dual_branch=True)
        self.d_block2 = EnhancedBlock3D(C[1], C[2], N[1], down=2, **cfgs)
        self.d_block3 = EnhancedBlock3D(C[2], C[3], N[2], down=2, **cfgs, ssm=True)
        self.d_block4 = EnhancedBlock3D(C[3], C[4], N[3], down=2, **cfgs, attn=True, ssm=True)

        self.u_block4 = EnhancedBlock3D(C[4], C[3], N[3], up=2, **cfgs, attn=True, ssm=True)
        self.u_block3 = EnhancedBlock3D(C[3]*2, C[2], N[2], up=2, **cfgs, ssm=True)
        self.u_block2 = EnhancedBlock3D(C[2]*2, C[1], N[1], up=2, **cfgs)
        self.u_block1 = EnhancedBlock3D(C[1]*2, C[0], N[0], **cfgs)
        
        self.out_conv = nn.Conv3d(C[0], self.out_channels, 3, 1, 1)
        self.out_conv.apply(ops.zero_out)

    def forward(self, 
                x_past: torch.Tensor, 
                x_furture: torch.Tensor, 
                egopose_past: torch.Tensor,
                egopose_future: torch.Tensor,
                timesteps: torch.Tensor
                ) -> torch.Tensor:
        
        # Merge inputs
        videos = torch.cat([x_past, x_furture], dim=1)
        h = videos.permute(0, 2, 1, 3, 4)
        B, C, F, H, W = h.shape

        # Merge pose information
        egopose = torch.cat([egopose_past, egopose_future], dim=1)  # [B, T, 3]
        pose_diff = compute_ego_pose_differences(egopose)


        # Timestep and pose embedding
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].repeat_interleave(h.shape[0], dim=0)
        temb = self.time_embedding(timesteps.to(h)) # B, 256

        # Pose embedding preserves temporal information
        pose_emb = self.pose_embedder(pose_diff)  # [B, T, temb_channels]

        # Expand time embedding to each timestep
        T = pose_emb.shape[1]
        temb_expanded = temb.unsqueeze(1).repeat(1, T, 1)  # [B, T, temb_channels]

        # Temporally-aware fusion
        fused_cond = torch.cat([temb_expanded, pose_emb], dim=-1)  # [B, T, temb_channels*2]
        temb_seq = self.fuse_proj(fused_cond)  # [B, T, temb_channels]

        # Spatial embedding
        if self.coords_encoding is not None:
            cenc = self.coords_encoding(self.coords)
            cenc = cenc.unsqueeze(2).repeat(B, 1, F, 1, 1)
            h = torch.cat([h, cenc], dim=1)

        h = self.in_conv(h) # B, 64, 10, 64, 256
        
        # Forward pass (carries branch features and temporal temb)
        h1, branch_feat1 = self.d_block1(h, temb_seq) # B, 64, 10, 64, 256
        h2, branch_feat2 = self.d_block2(h1, temb_seq, branch_feat1) # B, 128, 10, 32, 128
        h3, branch_feat3 = self.d_block3(h2, temb_seq, branch_feat2) # B, 256, 10, 16, 64
        h4, branch_feat4 = self.d_block4(h3, temb_seq, branch_feat3) # B, 512, 10, 8, 32
        
        h, _ = self.u_block4(h4, temb_seq, branch_feat4) # B, 256, 10, 16, 64
        h, _ = self.u_block3(_join(h, h3), temb_seq, branch_feat3) # B, 128, 10, 32, 128
        h, _ = self.u_block2(_join(h, h2), temb_seq, branch_feat2) # B, 64, 10, 64, 256
        h, _ = self.u_block1(_join(h, h1), temb_seq, branch_feat1) # B, 64, 10, 64, 256
        
        h = self.out_conv(h) # B, 3, 10, 32, 256
        
        return h.permute(0, 2, 1, 3, 4)