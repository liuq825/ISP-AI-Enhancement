from __future__ import annotations

import math

import torch
from torch import Tensor


def mse(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    error = (prediction - target).square()
    if mask is None:
        return error.mean()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=error.device, dtype=error.dtype)
    denominator = mask.expand_as(error).sum().clamp_min(1.0)
    return (error * mask).sum() / denominator


def psnr(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    data_range: float = 1.0,
) -> Tensor:
    error = mse(prediction, target, mask).clamp_min(torch.finfo(prediction.dtype).eps)
    return 10 * torch.log10(torch.tensor(data_range**2, device=error.device) / error)


def psnr_drop(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("PSNR values must be finite")
    return reference - candidate
