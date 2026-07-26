"""RAW 画质基础指标，支持有效区 mask 与剪枝前后 PSNR 差值计算。"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def mse(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """计算逐通道均方误差；mask 会广播到四个 RAW 通道。"""

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
    """由带 mask 的 MSE 计算 PSNR，并使用 dtype epsilon 避免无穷值。"""

    error = mse(prediction, target, mask).clamp_min(torch.finfo(prediction.dtype).eps)
    return 10 * torch.log10(torch.tensor(data_range**2, device=error.device) / error)


def psnr_drop(reference: float, candidate: float) -> float:
    """返回候选模型相对参考模型的 PSNR 下降量，并拒绝 NaN/Infinity。"""

    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("PSNR values must be finite")
    return reference - candidate
