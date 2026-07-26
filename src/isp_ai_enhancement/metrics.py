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
    """计算批内逐样本 PSNR 的算术平均，避免批大小影响统计权重。"""

    return psnr_per_sample(prediction, target, mask, data_range).mean()


def psnr_per_sample(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    data_range: float = 1.0,
) -> Tensor:
    """返回形状为 ``N`` 的逐样本 PSNR，可用于 Sensor/ISO 分桶统计。

    mask 接受 ``N×1×H×W`` 或 ``N×H×W`` 并广播到 RAW 四通道。每个样本独立
    归一化，防止最后一个不足 batch 的批次或不同有效面积改变样本权重。
    """

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same batched shape")
    error = (prediction - target).square()
    reduce_dimensions = tuple(range(1, error.ndim))
    if mask is None:
        per_sample_mse = error.mean(dim=reduce_dimensions)
    else:
        if mask.ndim == prediction.ndim - 1:
            mask = mask.unsqueeze(1)
        try:
            expanded_mask = mask.to(
                device=error.device, dtype=error.dtype
            ).expand_as(error)
        except RuntimeError as runtime_error:
            raise ValueError("mask cannot be broadcast to prediction shape") from runtime_error
        denominator = expanded_mask.sum(dim=reduce_dimensions).clamp_min(1.0)
        per_sample_mse = (error * expanded_mask).sum(dim=reduce_dimensions) / denominator
    per_sample_mse = per_sample_mse.clamp_min(torch.finfo(prediction.dtype).eps)
    peak = torch.tensor(data_range**2, device=error.device, dtype=error.dtype)
    return 10 * torch.log10(peak / per_sample_mse)


def psnr_drop(reference: float, candidate: float) -> float:
    """返回候选模型相对参考模型的 PSNR 下降量，并拒绝 NaN/Infinity。"""

    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("PSNR values must be finite")
    return reference - candidate
