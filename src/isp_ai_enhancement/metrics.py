"""RAW 画质基础指标，支持有效区 mask 与剪枝前后 PSNR 差值计算。"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


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


def ssim_per_sample(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """计算逐样本 packed RAW 高斯窗 SSIM，返回形状为 ``N`` 的张量。

    四个 canonical RAW 通道分别计算局部统计后等权平均。为避免边界填充影响，
    使用 valid 卷积；提供 mask 时，只有整个高斯窗口均有效的位置参与统计。
    该内部指标不宣称与 SIDD 服务器在 Bayer mosaic 上的实现逐位相同。
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM requires equal N×C×H×W prediction and target")
    if data_range <= 0 or sigma <= 0 or window_size <= 0:
        raise ValueError("SSIM data_range, sigma, and window_size must be positive")
    actual_window = min(window_size, prediction.shape[-2], prediction.shape[-1])
    if actual_window % 2 == 0:
        actual_window -= 1
    if actual_window <= 0:
        raise ValueError("SSIM input spatial dimensions are too small")
    coordinates = torch.arange(
        actual_window,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    coordinates = coordinates - (actual_window - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel_2d = torch.outer(gaussian, gaussian)
    channels = prediction.shape[1]
    kernel = kernel_2d.expand(channels, 1, actual_window, actual_window)

    mu_prediction = F.conv2d(prediction, kernel, groups=channels)
    mu_target = F.conv2d(target, kernel, groups=channels)
    mu_prediction_square = mu_prediction.square()
    mu_target_square = mu_target.square()
    mu_product = mu_prediction * mu_target
    variance_prediction = (
        F.conv2d(prediction.square(), kernel, groups=channels)
        - mu_prediction_square
    ).clamp_min(0)
    variance_target = (
        F.conv2d(target.square(), kernel, groups=channels) - mu_target_square
    ).clamp_min(0)
    covariance = (
        F.conv2d(prediction * target, kernel, groups=channels) - mu_product
    )
    constant_1 = (0.01 * data_range) ** 2
    constant_2 = (0.03 * data_range) ** 2
    score = (
        (2 * mu_product + constant_1)
        * (2 * covariance + constant_2)
        / (
            (mu_prediction_square + mu_target_square + constant_1)
            * (variance_prediction + variance_target + constant_2)
        ).clamp_min(torch.finfo(prediction.dtype).eps)
    )
    if mask is None:
        return score.mean(dim=(1, 2, 3))
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[0] != prediction.shape[0]:
        raise ValueError("SSIM mask must have shape N×1×H×W or N×H×W")
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    if mask.shape[1] != 1:
        raise ValueError("SSIM mask must contain one channel")
    # avg_pool 等于 1 才表示整个局部窗口有效；边界和 Pad 区不会污染 SSIM。
    valid_window = (
        F.avg_pool2d(mask, kernel_size=actual_window, stride=1) >= 1.0 - 1e-6
    ).to(prediction.dtype)
    valid_window = valid_window.expand_as(score)
    denominator = valid_window.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (score * valid_window).sum(dim=(1, 2, 3)) / denominator


def psnr_drop(reference: float, candidate: float) -> float:
    """返回候选模型相对参考模型的 PSNR 下降量，并拒绝 NaN/Infinity。"""

    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("PSNR values must be finite")
    return reference - candidate
