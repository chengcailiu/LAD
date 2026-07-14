from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .LAD_UNETR import (
        AnisotropicRoPE,
        DropPath,
        FeedForward,
        HeadwiseAnisotropyLearner,
        SegmentationHead,
        _apply_headwise_channel_weights,
        _build_grid,
        _to_3tuple,
    )
except ImportError:
    from LAD_UNETR import (
        AnisotropicRoPE,
        DropPath,
        FeedForward,
        HeadwiseAnisotropyLearner,
        SegmentationHead,
        _apply_headwise_channel_weights,
        _build_grid,
        _to_3tuple,
    )


def _window_partition(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    b, d, h, w, c = x.shape
    wd, wh, ww = window_size
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, wd, wh, ww, c)


def _window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    batch_size: int,
    padded_size: tuple[int, int, int],
) -> torch.Tensor:
    dp, hp, wp = padded_size
    wd, wh, ww = window_size
    x = windows.view(batch_size, dp // wd, hp // wh, wp // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch_size, dp, hp, wp, -1)


def _pad_to_window(
    x: torch.Tensor,
    window_size: tuple[int, int, int],
    value: float = 0.0,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    d, h, w = x.shape[1:4]
    pad_d = (window_size[0] - d % window_size[0]) % window_size[0]
    pad_h = (window_size[1] - h % window_size[1]) % window_size[1]
    pad_w = (window_size[2] - w % window_size[2]) % window_size[2]
    if pad_d or pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d), value=value)
    return x, (d + pad_d, h + pad_h, w + pad_w)


def _compute_shift_mask(
    padded_size: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not any(shift_size):
        return None

    dp, hp, wp = padded_size
    wd, wh, ww = window_size
    sd, sh, sw = shift_size
    img_mask = torch.zeros((1, dp, hp, wp, 1), device=device)

    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)) if sd > 0 else (slice(0, dp),)
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)) if sh > 0 else (slice(0, hp),)
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)) if sw > 0 else (slice(0, wp),)

    region_id = 0
    for d_slice in d_slices:
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, d_slice, h_slice, w_slice, :] = region_id
                region_id += 1

    mask_windows = _window_partition(img_mask, window_size).view(-1, wd * wh * ww)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int | Sequence[int]) -> None:
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 4, 1).contiguous()
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim * 8)
        self.reduction = nn.Linear(dim * 8, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w, c = x.shape
        pad_d = d % 2
        pad_h = h % 2
        pad_w = w % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        parts = [
            x[:, i::2, j::2, k::2, :]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ]
        x = torch.cat(parts, dim=-1)
        return self.reduction(self.norm(x))


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int, int],
        dropout: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.dropout = float(dropout)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = AnisotropicRoPE(self.head_dim) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        scales: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self.rope is not None:
            q = self.rope(q, grid, scales)
            k = self.rope(k, grid, scales)

        if attn_mask is None and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                num_windows = attn_mask.shape[0]
                batch_size = b_windows // num_windows
                attn = attn.view(batch_size, num_windows, self.num_heads, n, n)
                attn = attn + attn_mask.view(1, num_windows, 1, n, n)
                attn = attn.view(-1, self.num_heads, n, n)
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b_windows, n, c)
        return self.proj_drop(self.proj(out))


class LADSwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int | Sequence[int],
        shift_size: int | Sequence[int] = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_rope: bool = True,
        use_aafm: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = _to_3tuple(window_size)
        self.shift_size = _to_3tuple(shift_size)
        self.use_aafm = use_aafm
        self.use_lad = use_rope or use_aafm
        self.norm1 = nn.LayerNorm(dim)
        self.anisotropy = HeadwiseAnisotropyLearner(dim, num_heads, use_aafm=use_aafm) if self.use_lad else None
        self.attn = WindowAttention(dim, num_heads, self.window_size, dropout, use_rope)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, return_scales: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        shortcut = x
        b, d, h, w, c = x.shape
        x_norm = self.norm1(x)

        scales = None
        channel_weights = None
        if self.anisotropy is not None:
            scales, channel_weights = self.anisotropy(x_norm.permute(0, 4, 1, 2, 3).contiguous())
        if self.use_aafm and channel_weights is not None:
            flat = x_norm.reshape(b, d * h * w, c)
            flat = _apply_headwise_channel_weights(flat, channel_weights, self.num_heads)
            x_norm = flat.reshape(b, d, h, w, c)

        grid = _build_grid((d, h, w), x.device).reshape(1, d, h, w, 3).expand(b, -1, -1, -1, -1)
        x_pad, padded_size = _pad_to_window(x_norm, self.window_size)
        grid_pad, _ = _pad_to_window(grid, self.window_size)
        shift_size = tuple(min(s, ws // 2) if dim > ws else 0 for s, ws, dim in zip(self.shift_size, self.window_size, padded_size))

        if any(shift_size):
            shifted_x = torch.roll(x_pad, shifts=tuple(-s for s in shift_size), dims=(1, 2, 3))
            shifted_grid = torch.roll(grid_pad, shifts=tuple(-s for s in shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x_pad
            shifted_grid = grid_pad

        x_windows = _window_partition(shifted_x, self.window_size).view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], c)
        grid_windows = _window_partition(shifted_grid, self.window_size).view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], 3)
        num_windows_per_sample = x_windows.shape[0] // b
        window_scales = scales.repeat_interleave(num_windows_per_sample, dim=0) if scales is not None else None
        attn_mask = _compute_shift_mask(padded_size, self.window_size, shift_size, x.device)

        attn_windows = self.attn(x_windows, grid_windows, window_scales, attn_mask)
        attn_windows = attn_windows.view(-1, *self.window_size, c)
        shifted_x = _window_reverse(attn_windows, self.window_size, b, padded_size)

        if any(shift_size):
            x_attn = torch.roll(shifted_x, shifts=shift_size, dims=(1, 2, 3))
        else:
            x_attn = shifted_x
        x_attn = x_attn[:, :d, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, scales if return_scales else None


class LADSwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int | Sequence[int],
        mlp_ratio: float,
        dropout: float,
        drop_path: Sequence[float],
        downsample: bool,
        use_rope: bool,
        use_aafm: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for index in range(depth):
            shift = tuple(ws // 2 for ws in _to_3tuple(window_size)) if index % 2 == 1 else (0, 0, 0)
            self.blocks.append(
                LADSwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=float(drop_path[index]),
                    use_rope=use_rope,
                    use_aafm=use_aafm,
                )
            )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, return_scales: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        scale_trace = []
        for block in self.blocks:
            x, scales = block(x, return_scales=return_scales)
            if return_scales:
                scale_trace.append(scales)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, scale_trace


class LAD_SwinUNETR(nn.Module):
    """
    Public model-only LAD implementation for a SwinUNETR-style backbone.

    Set task="segmentation" or task="classification". LAD scales are learned
    per instance and per attention head, then reused by windowed RoPE and AAFM.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        patch_size: int | Sequence[int] = 2,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int | Sequence[int] = 7,
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
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length.")
        if task == "segmentation" and out_channels is None:
            raise ValueError("out_channels is required for segmentation.")
        if task == "classification" and num_classes is None:
            raise ValueError("num_classes is required for classification.")

        self.task = task
        self.patch_embed = PatchEmbed3D(in_channels, feature_size, patch_size)

        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()
        cursor = 0
        dim = feature_size
        self.stages = nn.ModuleList()
        for stage_index, depth in enumerate(depths):
            stage_drop = dpr[cursor : cursor + depth]
            cursor += depth
            downsample = stage_index < len(depths) - 1
            self.stages.append(
                LADSwinStage(
                    dim=dim,
                    depth=depth,
                    num_heads=num_heads[stage_index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=stage_drop,
                    downsample=downsample,
                    use_rope=use_rope,
                    use_aafm=use_aafm,
                )
            )
            if downsample:
                dim *= 2

        self.out_dim = dim
        self.norm = nn.LayerNorm(dim)
        if task == "segmentation":
            self.head = SegmentationHead(dim, decoder_channels, int(out_channels))
        else:
            self.head = nn.Linear(dim, int(num_classes))

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

    def forward(self, x: torch.Tensor, return_scales: bool = False):
        input_size = tuple(int(v) for v in x.shape[2:])
        x = self.patch_embed(x)

        scale_trace = []
        for stage in self.stages:
            x, stage_scales = stage(x, return_scales=return_scales)
            if return_scales:
                scale_trace.extend(stage_scales)

        x = self.norm(x)
        if self.task == "classification":
            output = self.head(x.mean(dim=(1, 2, 3)))
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            output = self.head(x, input_size)

        if return_scales:
            return output, scale_trace
        return output


__all__ = ["LAD_SwinUNETR"]
