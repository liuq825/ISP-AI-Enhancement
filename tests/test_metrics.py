"""验证逐样本 PSNR 的批次权重和有效区域掩码语义。"""

import torch

from isp_ai_enhancement.metrics import psnr, psnr_per_sample, ssim_per_sample


def test_psnr_returns_equal_weight_per_sample_average() -> None:
    """不同误差样本应先各算 PSNR 再平均，而不是先混合 MSE。"""

    target = torch.zeros(2, 4, 2, 2)
    prediction = torch.stack(
        (torch.full((4, 2, 2), 0.1), torch.full((4, 2, 2), 0.01))
    )
    values = psnr_per_sample(prediction, target)
    torch.testing.assert_close(values, torch.tensor([20.0, 40.0]))
    torch.testing.assert_close(psnr(prediction, target), torch.tensor(30.0))


def test_psnr_per_sample_applies_mask_independently() -> None:
    """每个样本的 mask 分母应独立计算，且广播到四个 RAW 通道。"""

    target = torch.zeros(2, 4, 2, 2)
    prediction = torch.ones_like(target)
    prediction[0, :, 0, 0] = 0.1
    mask = torch.zeros(2, 1, 2, 2)
    mask[:, :, 0, 0] = 1
    values = psnr_per_sample(prediction, target, mask)
    torch.testing.assert_close(values, torch.tensor([20.0, 0.0]), atol=1e-5, rtol=1e-5)


def test_ssim_is_one_for_identical_packed_raw_and_uses_mask() -> None:
    """相同 RAW 的 SSIM 必须为 1，mask 外差异不能改变结果。"""

    target = torch.rand(1, 4, 16, 16)
    prediction = target.clone()
    mask = torch.ones(1, 1, 16, 16)
    mask[:, :, :8] = 0
    prediction[:, :, :4] = 0
    score = ssim_per_sample(prediction, target, mask, window_size=5)
    torch.testing.assert_close(score, torch.ones(1), atol=1e-5, rtol=1e-5)
