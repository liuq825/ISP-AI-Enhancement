from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TileConfig:
    size: int = 512
    halo: int = 48
    multiple: int = 16
    min_weight: float = 1e-3

    def validate(self) -> None:
        if self.size <= 0 or self.size % self.multiple:
            raise ValueError("tile size must be a positive multiple of model alignment")
        if self.halo < 0 or 2 * self.halo >= self.size:
            raise ValueError("halo must satisfy 0 <= 2*halo < tile size")


@dataclass(frozen=True)
class TiledResult:
    residual: Tensor
    tile_count: int
    runtime_ms: float


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _blend_window(size: int, min_weight: float, device: torch.device, dtype) -> Tensor:
    # Keep the two-dimensional outer product above min_weight at the borders.
    edge_floor = min_weight**0.5
    one_dimensional = torch.hann_window(size, periodic=False, device=device, dtype=dtype).clamp_min(
        edge_floor
    )
    return torch.outer(one_dimensional, one_dimensional).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def tiled_inference(
    model: nn.Module,
    context: Tensor,
    config: TileConfig | None = None,
) -> TiledResult:
    """Run overlap-add tile inference while maintaining the valid-mask channel."""
    tile = config or TileConfig()
    tile.validate()
    if context.ndim != 4 or context.shape[1] != 16:
        raise ValueError("context must have shape N×16×H×W")
    original_height, original_width = context.shape[-2:]
    padded_height = max(tile.size, original_height)
    padded_width = max(tile.size, original_width)
    pad_h = padded_height - original_height
    pad_w = padded_width - original_width
    padded = F.pad(context, (0, pad_w, 0, pad_h), value=0.0)
    padded[:, 15:16, :original_height, :original_width] = context[:, 15:16]
    if pad_h or pad_w:
        padded[:, 15:16, original_height:, :] = 0
        padded[:, 15:16, :, original_width:] = 0

    stride = tile.size - 2 * tile.halo
    y_starts = _starts(padded_height, tile.size, stride)
    x_starts = _starts(padded_width, tile.size, stride)
    output = torch.zeros(
        context.shape[0],
        4,
        padded_height,
        padded_width,
        device=context.device,
        dtype=context.dtype,
    )
    weight_sum = torch.zeros_like(output[:, :1])
    window = _blend_window(tile.size, tile.min_weight, context.device, context.dtype)
    started = perf_counter()
    count = 0
    for top in y_starts:
        for left in x_starts:
            patch = padded[..., top : top + tile.size, left : left + tile.size]
            residual = model(patch)
            if residual.shape != (context.shape[0], 4, tile.size, tile.size):
                raise ValueError(f"model returned invalid tile shape {tuple(residual.shape)}")
            output[..., top : top + tile.size, left : left + tile.size] += residual * window
            weight_sum[..., top : top + tile.size, left : left + tile.size] += window
            count += 1
    output = output / weight_sum.clamp_min(tile.min_weight)
    runtime_ms = (perf_counter() - started) * 1000
    return TiledResult(
        residual=output[..., :original_height, :original_width],
        tile_count=count,
        runtime_ms=runtime_ms,
    )
