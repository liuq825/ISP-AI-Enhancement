"""验证 QAT 模块替换范围和观察器开关。"""

import torch

from isp_ai_enhancement.models.nafnet import NAFNetRaw
from isp_ai_enhancement.quantization.fake_quant import (
    QATConv2d,
    prepare_qat,
    set_observer_enabled,
)


def test_prepare_qat_excludes_intro_and_ending() -> None:
    """首尾卷积保持高精度，其余卷积应替换为可训练伪量化版本。"""

    model = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    )
    report = prepare_qat(model)
    assert report.converted_convolutions > 0
    assert not isinstance(model.intro, QATConv2d)
    assert not isinstance(model.ending, QATConv2d)
    assert isinstance(model.encoders[0][0].conv1, QATConv2d)
    value = torch.randn(1, 16, 16, 16)
    assert model(value).shape == (1, 4, 16, 16)
    set_observer_enabled(model, False)
