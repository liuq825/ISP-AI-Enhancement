"""验证 QAT 模块替换范围和观察器开关。"""

import torch

from isp_ai_enhancement.models.nafnet import NAFNetRaw
from isp_ai_enhancement.quantization.fake_quant import (
    QATConv2d,
    SymmetricFakeQuant,
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
    assert report.simulated_int8_weight_ratio > 0.9
    value = torch.randn(1, 16, 16, 16)
    assert model(value).shape == (1, 4, 16, 16)
    set_observer_enabled(model, False)


def test_fake_quant_observer_initializes_from_first_real_batch() -> None:
    """首批低幅 RAW 应直接初始化范围，不能从默认 1.0 缓慢衰减。"""

    quantizer = SymmetricFakeQuant(bits=8, momentum=0.95)
    value = torch.full((1, 4, 8, 8), 0.125)
    quantizer(value)
    torch.testing.assert_close(quantizer.max_abs, torch.tensor([0.125]))
    assert bool(quantizer.observer_initialized.item())

    set_observer_enabled(quantizer, False)
    quantizer(torch.ones_like(value))
    torch.testing.assert_close(quantizer.max_abs, torch.tensor([0.125]))


def test_fake_quant_observer_does_not_update_during_evaluation() -> None:
    """模型 eval 阶段应复用训练尺度，不能被验证集悄然改写。"""

    quantizer = SymmetricFakeQuant(bits=8, momentum=0.5)
    quantizer(torch.full((1, 1, 2, 2), 0.25))
    quantizer.eval()
    quantizer(torch.ones(1, 1, 2, 2))
    torch.testing.assert_close(quantizer.max_abs, torch.tensor([0.25]))


def test_qat_weight_observer_shape_is_checkpoint_stable() -> None:
    """权重观察器应在首批前按输出通道定形，保证 strict checkpoint 恢复。"""

    source = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    )
    target = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    )
    prepare_qat(source)
    prepare_qat(target)
    source.train()
    source(torch.randn(1, 16, 16, 16))

    target.load_state_dict(source.state_dict(), strict=True)
    source_scale = source.encoders[0][0].conv1.weight_fake_quant.max_abs
    target_scale = target.encoders[0][0].conv1.weight_fake_quant.max_abs
    assert source_scale.shape == (8,)
    torch.testing.assert_close(target_scale, source_scale)
