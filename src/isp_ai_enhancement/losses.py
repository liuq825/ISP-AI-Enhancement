"""RAW 域监督训练损失：稳健像素、空间梯度、色比和 Teacher 输出蒸馏。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _masked_mean(value: Tensor, mask: Tensor | None) -> Tensor:
    """只对有效像素求均值，分母最小为 1 以避免全零 mask 产生 NaN。"""

    if mask is None:
        return value.mean()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=value.device, dtype=value.dtype)
    weighted = value * mask
    denominator = mask.expand_as(value).sum().clamp_min(1.0)
    return weighted.sum() / denominator


def charbonnier_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    epsilon: float = 1e-3,
) -> Tensor:
    """计算平滑 L1 风格的 Charbonnier 损失，降低离群坏点对训练的影响。"""

    return _masked_mean(torch.sqrt((prediction - target).square() + epsilon**2), mask)


def gradient_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """约束水平/垂直一阶差分，减少过度平滑并保留文字、毛发和边缘。"""

    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_x = None if mask is None else mask[..., :, 1:] * mask[..., :, :-1]
    mask_y = None if mask is None else mask[..., 1:, :] * mask[..., :-1, :]
    return _masked_mean((pred_dx - target_dx).abs(), mask_x) + _masked_mean(
        (pred_dy - target_dy).abs(), mask_y
    )


def color_ratio_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """约束 `[R,Gr,Gb,B]` 相对绿色均值的比例，抑制 RAW 域色偏漂移。"""

    if prediction.shape[1] != 4 or target.shape[1] != 4:
        raise ValueError("color ratio loss requires four packed RAW channels")
    if mask is None:
        pred_mean = prediction.mean(dim=(-2, -1))
        target_mean = target.mean(dim=(-2, -1))
    else:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        denominator = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        pred_mean = (prediction * mask).sum(dim=(-2, -1)) / denominator
        target_mean = (target * mask).sum(dim=(-2, -1)) / denominator
    pred_ratio = pred_mean / pred_mean[:, 1:3].mean(dim=1, keepdim=True).clamp_min(1e-4)
    target_ratio = target_mean / target_mean[:, 1:3].mean(dim=1, keepdim=True).clamp_min(1e-4)
    return F.smooth_l1_loss(pred_ratio, target_ratio)


@dataclass(frozen=True)
class LossWeights:
    """集中保存各损失项权重，便于把完整配方写入 checkpoint。"""

    charbonnier: float = 1.0
    gradient: float = 0.1
    color: float = 0.05
    teacher_output: float = 0.0


class RawRestorationLoss(nn.Module):
    """组合 RAW 重建与可选 Teacher 输出蒸馏损失，并返回逐项日志。"""

    def __init__(self, weights: LossWeights | None = None) -> None:
        """保存不可变损失权重；未提供时使用保守基线值。"""

        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        enhanced: Tensor,
        target: Tensor,
        *,
        mask: Tensor | None = None,
        teacher_enhanced: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """计算加权总损失和未加权分项，Teacher 张量会先停止梯度。"""

        terms = {
            "charbonnier": charbonnier_loss(enhanced, target, mask),
            "gradient": gradient_loss(enhanced, target, mask),
            "color": color_ratio_loss(enhanced, target, mask),
        }
        if teacher_enhanced is not None and self.weights.teacher_output > 0:
            terms["teacher_output"] = charbonnier_loss(enhanced, teacher_enhanced.detach(), mask)
        total = (
            self.weights.charbonnier * terms["charbonnier"]
            + self.weights.gradient * terms["gradient"]
            + self.weights.color * terms["color"]
        )
        if "teacher_output" in terms:
            total = total + self.weights.teacher_output * terms["teacher_output"]
        return total, terms
