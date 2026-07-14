from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3tuple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("Expected an int or a 3-item sequence.")
    return tuple(int(v) for v in value)


def _build_grid(grid_size: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    axes = [torch.arange(size, device=device, dtype=torch.float32) for size in grid_size]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid.reshape(1, -1, 3)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleAxisLearner(nn.Module):
    """Regress per-head scale logits from a one-dimensional axis profile."""

    def __init__(self, channels: int, num_heads: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.profile_encoder = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, num_heads),
        )
        self.regressor[-1]._lad_zero_init = True
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, axis_profile: torch.Tensor) -> torch.Tensor:
        features = self.profile_encoder(axis_profile).mean(dim=-1)
        return self.regressor(features)


class HeadwiseAnisotropyLearner(nn.Module):
    """
    Instance-aware anisotropy learner from LAD.

    The output scale tensor has shape [B, num_heads, 3], matching the paper's
    head-wise depth, height, and width scales.
    """

    def __init__(self, channels: int, num_heads: int, reduction: int = 4, use_aafm: bool = True) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_aafm = use_aafm

        self.depth_learner = SingleAxisLearner(channels, num_heads, reduction)
        self.height_learner = SingleAxisLearner(channels, num_heads, reduction)
        self.width_learner = SingleAxisLearner(channels, num_heads, reduction)

        if use_aafm:
            self.gate_mlp = nn.Sequential(
                nn.Linear(3, 12),
                nn.GELU(),
                nn.Linear(12, 3),
            )
            self.gate_mlp[-1]._lad_zero_init = True
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
        else:
            self.gate_mlp = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError("Expected a channel-first 3D tensor shaped [B, C, D, H, W].")

        depth_logits = self.depth_learner(x.mean(dim=(3, 4)))
        height_logits = self.height_learner(x.mean(dim=(2, 4)))
        width_logits = self.width_learner(x.mean(dim=(2, 3)))
        raw_logits = torch.stack((depth_logits, height_logits, width_logits), dim=-1)
        scales = 1.0 + torch.tanh(raw_logits)

        if self.gate_mlp is None:
            return scales, None

        gates = 1.0 + torch.tanh(self.gate_mlp(scales))
        channel_weights = torch.ones(
            x.shape[0],
            self.num_heads,
            self.head_dim,
            dtype=x.dtype,
            device=x.device,
        )
        pairs_per_axis = (self.head_dim // 3) // 2
        axis_width = pairs_per_axis * 2
        for axis in range(3):
            start = axis * axis_width
            end = start + axis_width
            if start < end:
                channel_weights[:, :, start:end] = gates[:, :, axis].unsqueeze(-1)
        return scales, channel_weights


def _apply_headwise_channel_weights(
    x: torch.Tensor,
    channel_weights: Optional[torch.Tensor],
    num_heads: int,
) -> torch.Tensor:
    if channel_weights is None:
        return x
    b, n, c = x.shape
    head_dim = c // num_heads
    x = x.reshape(b, n, num_heads, head_dim)
    x = x * channel_weights.unsqueeze(1)
    return x.reshape(b, n, c)


class AnisotropicRoPE(nn.Module):
    """Frequency-modulated anisotropic RoPE with per-head spatial scales."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        pairs_per_axis = (head_dim // 3) // 2
        axis_width = pairs_per_axis * 2
        self.head_dim = head_dim
        self.pairs_per_axis = pairs_per_axis
        self.axis_width = axis_width
        self.rotary_dim = axis_width * 3
        if pairs_per_axis > 0:
            inv_freq = 1.0 / (base ** (torch.arange(0, axis_width, 2).float() / axis_width))
        else:
            inv_freq = torch.empty(0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, grid: torch.Tensor, scales: Optional[torch.Tensor]) -> torch.Tensor:
        if self.rotary_dim == 0:
            return x
        if scales is None:
            scales = torch.ones(x.shape[0], x.shape[1], 3, dtype=x.dtype, device=x.device)

        rotated_parts = []
        for axis in range(3):
            start = axis * self.axis_width
            end = start + self.axis_width
            part = x[..., start:end]
            theta = (
                grid[:, None, :, axis, None].to(dtype=torch.float32)
                * scales[:, :, None, axis, None].to(dtype=torch.float32)
                * self.inv_freq[None, None, None, :].to(dtype=torch.float32)
            )
            rotated_parts.append(self._rotate(part, theta))

        if self.rotary_dim < self.head_dim:
            rotated_parts.append(x[..., self.rotary_dim :])
        return torch.cat(rotated_parts, dim=-1)

    @staticmethod
    def _rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(dtype=torch.float32).reshape(*x.shape[:-1], -1, 2)
        x_real, x_imag = x.unbind(dim=-1)
        cos = theta.cos()
        sin = theta.sin()
        out = torch.stack((x_real * cos - x_imag * sin, x_real * sin + x_imag * cos), dim=-1)
        return out.flatten(-2).to(dtype=dtype)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    training: bool,
    scale: float,
) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p if training else 0.0,
            scale=scale,
        )

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=dropout_p, training=training)
    return torch.matmul(attn, v)


class LADTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: tuple[int, int, int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_rope: bool = True,
        use_aafm: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid_size = grid_size
        self.use_rope = use_rope
        self.use_aafm = use_aafm
        self.use_lad = use_rope or use_aafm

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = float(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        self.rope = AnisotropicRoPE(self.head_dim) if use_rope else None
        self.anisotropy = HeadwiseAnisotropyLearner(dim, num_heads, use_aafm=use_aafm) if self.use_lad else None
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        return_scales: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, n, c = x.shape
        x_norm = self.norm1(x)
        scales = None
        channel_weights = None
        if self.anisotropy is not None:
            volume = x_norm.transpose(1, 2).reshape(b, c, *self.grid_size)
            scales, channel_weights = self.anisotropy(volume)
        x_attn = _apply_headwise_channel_weights(x_norm, channel_weights if self.use_aafm else None, self.num_heads)

        qkv = self.qkv(x_attn).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        if self.rope is not None:
            q = self.rope(q, grid, scales)
            k = self.rope(k, grid, scales)

        attn_out = _attention(q, k, v, self.attn_drop, self.training, self.scale)
        attn_out = attn_out.transpose(1, 2).reshape(b, n, c)
        x = x + self.drop_path(self.proj_drop(self.proj(attn_out)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, scales if return_scales else None


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int | Sequence[int]) -> None:
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)
        grid_size = tuple(int(v) for v in x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
        )
        self.out = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int, int]) -> torch.Tensor:
        x = self.proj(x)
        x = F.interpolate(x, size=output_size, mode="trilinear", align_corners=False)
        return self.out(x)


class LAD_UNETR(nn.Module):
    """
    Public model-only LAD implementation for a UNETR-style ViT backbone.

    Set task="segmentation" for dense prediction or task="classification" for
    volume-level prediction. Learned scales are per-instance and per-head.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        img_size: int | Sequence[int] = (96, 96, 96),
        patch_size: int | Sequence[int] = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        decoder_channels: int = 96,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        task: str = "segmentation",
        use_rope: bool = True,
        use_aafm: bool = True,
    ) -> None:
        super().__init__()
        if task not in {"segmentation", "classification"}:
            raise ValueError('task must be "segmentation" or "classification".')
        if task == "segmentation" and out_channels is None:
            raise ValueError("out_channels is required for segmentation.")
        if task == "classification" and num_classes is None:
            raise ValueError("num_classes is required for classification.")

        self.task = task
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        self.img_size = _to_3tuple(img_size)
        self.patch_size = _to_3tuple(patch_size)
        self.base_grid_size = tuple(i // p for i, p in zip(self.img_size, self.patch_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, math.prod(self.base_grid_size), embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                LADTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    grid_size=self.base_grid_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                    use_rope=use_rope,
                    use_aafm=use_aafm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if task == "segmentation":
            self.head = SegmentationHead(embed_dim, decoder_channels, int(out_channels))
        else:
            self.head = nn.Linear(embed_dim, int(num_classes))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if getattr(module, "_lad_zero_init", False):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _position_embedding(self, grid_size: tuple[int, int, int]) -> torch.Tensor:
        if grid_size == self.base_grid_size:
            return self.pos_embed
        pos = self.pos_embed.reshape(1, *self.base_grid_size, self.embed_dim).permute(0, 4, 1, 2, 3)
        pos = F.interpolate(pos, size=grid_size, mode="trilinear", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, return_scales: bool = False):
        input_size = tuple(int(v) for v in x.shape[2:])
        tokens, grid_size = self.patch_embed(x)
        if any(size <= 0 for size in grid_size):
            raise ValueError("Input spatial size is too small for the configured patch size.")

        grid = _build_grid(grid_size, tokens.device).expand(tokens.shape[0], -1, -1)
        tokens = self.pos_drop(tokens + self._position_embedding(grid_size).to(dtype=tokens.dtype, device=tokens.device))

        scale_trace = []
        for block in self.blocks:
            block.grid_size = grid_size
            tokens, scales = block(tokens, grid, return_scales=return_scales)
            if return_scales:
                scale_trace.append(scales)

        tokens = self.norm(tokens)
        if self.task == "classification":
            output = self.head(tokens.mean(dim=1))
        else:
            b, _, c = tokens.shape
            volume = tokens.transpose(1, 2).reshape(b, c, *grid_size)
            output = self.head(volume, input_size)

        if return_scales:
            return output, scale_trace
        return output


__all__ = ["LAD_UNETR"]
