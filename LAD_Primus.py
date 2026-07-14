from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .LAD_UNETR import (
        AnisotropicRoPE,
        DropPath,
        HeadwiseAnisotropyLearner,
        PatchEmbed3D,
        SegmentationHead,
        _apply_headwise_channel_weights,
        _attention,
        _build_grid,
        _to_3tuple,
    )
except ImportError:
    from LAD_UNETR import (
        AnisotropicRoPE,
        DropPath,
        HeadwiseAnisotropyLearner,
        PatchEmbed3D,
        SegmentationHead,
        _apply_headwise_channel_weights,
        _attention,
        _build_grid,
        _to_3tuple,
    )


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc1(x).chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class LADEvaAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = AnisotropicRoPE(self.head_dim) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        scales: torch.Tensor,
        num_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.rope is not None:
            if num_prefix_tokens > 0:
                q_prefix, q_spatial = q[:, :, :num_prefix_tokens], q[:, :, num_prefix_tokens:]
                k_prefix, k_spatial = k[:, :, :num_prefix_tokens], k[:, :, num_prefix_tokens:]
                q_spatial = self.rope(q_spatial, grid, scales)
                k_spatial = self.rope(k_spatial, grid, scales)
                q = torch.cat((q_prefix, q_spatial), dim=2)
                k = torch.cat((k_prefix, k_spatial), dim=2)
            else:
                q = self.rope(q, grid, scales)
                k = self.rope(k, grid, scales)

        x = _attention(q, k, v, self.dropout, self.training, self.scale)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class LADEvaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: tuple[int, int, int],
        mlp_ratio: float = 8.0 / 3.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = 0.1,
        num_prefix_tokens: int = 0,
        use_rope: bool = True,
        use_aafm: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.num_prefix_tokens = num_prefix_tokens
        self.use_aafm = use_aafm
        self.use_lad = use_rope or use_aafm

        self.norm1 = nn.LayerNorm(dim)
        self.anisotropy = HeadwiseAnisotropyLearner(dim, num_heads, use_aafm=use_aafm) if self.use_lad else None
        self.attn = LADEvaAttention(dim, num_heads, dropout, use_rope)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout)

        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        return_scales: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, _, c = x.shape
        x_norm = self.norm1(x)
        scales = None
        channel_weights = None
        if self.anisotropy is not None:
            spatial_tokens = x_norm[:, self.num_prefix_tokens :] if self.num_prefix_tokens else x_norm
            volume = spatial_tokens.transpose(1, 2).reshape(b, c, *self.grid_size)
            scales, channel_weights = self.anisotropy(volume)

        x_attn = x_norm
        if self.use_aafm and channel_weights is not None:
            x_attn = _apply_headwise_channel_weights(x_attn, channel_weights, self.num_heads)

        attn_out = self.attn(x_attn, grid, scales, self.num_prefix_tokens)
        if self.gamma1 is not None:
            attn_out = self.gamma1 * attn_out
        x = x + self.drop_path(attn_out)

        mlp_out = self.mlp(self.norm2(x))
        if self.gamma2 is not None:
            mlp_out = self.gamma2 * mlp_out
        x = x + self.drop_path(mlp_out)
        return x, scales if return_scales else None


class LAD_Primus(nn.Module):
    """
    Public model-only LAD implementation for a Primus/EVA-02-style backbone.

    Set task="segmentation" or task="classification". The LAD learner returns
    per-instance, per-head scales aligned with anisotropic RoPE and AAFM.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        img_size: int | Sequence[int] = (96, 96, 96),
        patch_size: int | Sequence[int] = (8, 8, 8),
        embed_dim: int = 792,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 8.0 / 3.0,
        decoder_channels: int = 96,
        num_register_tokens: int = 0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 0.1,
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
        self.num_register_tokens = int(num_register_tokens)
        self.img_size = _to_3tuple(img_size)
        self.patch_size = _to_3tuple(patch_size)
        self.base_grid_size = tuple(i // p for i, p in zip(self.img_size, self.patch_size))
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, self.patch_size)

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, math.prod(self.base_grid_size), embed_dim))
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim))
            self.register_pos_embed = nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim))
        else:
            self.register_tokens = None
            self.register_pos_embed = None
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                LADEvaBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    grid_size=self.base_grid_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                    init_values=init_values,
                    num_prefix_tokens=self.num_register_tokens,
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

        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
            nn.init.trunc_normal_(self.register_pos_embed, std=0.02)
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
            return self.spatial_pos_embed
        pos = self.spatial_pos_embed.reshape(1, *self.base_grid_size, self.embed_dim).permute(0, 4, 1, 2, 3)
        pos = F.interpolate(pos, size=grid_size, mode="trilinear", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, return_scales: bool = False):
        input_size = tuple(int(v) for v in x.shape[2:])
        tokens, grid_size = self.patch_embed(x)
        spatial_pos = self._position_embedding(grid_size).to(dtype=tokens.dtype, device=tokens.device)
        tokens = tokens + spatial_pos

        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(tokens.shape[0], -1, -1)
            register_pos = self.register_pos_embed.to(dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat((register_tokens + register_pos, tokens), dim=1)

        tokens = self.pos_drop(tokens)
        grid = _build_grid(grid_size, tokens.device).expand(tokens.shape[0], -1, -1)

        scale_trace = []
        for block in self.blocks:
            block.grid_size = grid_size
            tokens, scales = block(tokens, grid, return_scales=return_scales)
            if return_scales:
                scale_trace.append(scales)

        tokens = self.norm(tokens)
        spatial_tokens = tokens[:, self.num_register_tokens :] if self.num_register_tokens else tokens
        if self.task == "classification":
            output = self.head(spatial_tokens.mean(dim=1))
        else:
            b, _, c = spatial_tokens.shape
            volume = spatial_tokens.transpose(1, 2).reshape(b, c, *grid_size)
            output = self.head(volume, input_size)

        if return_scales:
            return output, scale_trace
        return output


__all__ = ["LAD_Primus"]
