"""验证 NAFNet RAW 模型形状、参数基线和静态导出路径。"""

import torch

from isp_ai_enhancement.models.nafnet import NAFNetRaw


def test_small_model_shape_and_padding() -> None:
    """非 16 倍数输入应自动补边并裁回原始空间尺寸。"""

    model = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    )
    value = torch.randn(1, 16, 17, 19)
    residual = model(value)
    assert residual.shape == (1, 4, 17, 19)
    assert model.enhance(value).min().item() >= 0
    assert model.enhance(value).max().item() <= 1


def test_reference_parameter_count() -> None:
    """升级后的 Student 应采用可配置 `[2,2,6,8]` 并锁定精确参数基线。"""

    model = NAFNetRaw()
    assert model.encoder_blocks == (2, 2, 6, 8)
    assert model.parameter_count() == 14_586_340


def test_static_path_matches_regular_path_for_aligned_input() -> None:
    """对齐尺寸下的 ONNX 静态路径应与常规前向数值一致。"""

    model = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    ).eval()
    value = torch.randn(1, 16, 32, 48)
    torch.testing.assert_close(model(value), model.forward_static(value))
