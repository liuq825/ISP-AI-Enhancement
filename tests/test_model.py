import torch

from isp_ai_enhancement.models.nafnet import NAFNetRaw


def test_small_model_shape_and_padding() -> None:
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
    model = NAFNetRaw()
    assert model.parameter_count() == 14_348_516


def test_static_path_matches_regular_path_for_aligned_input() -> None:
    model = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    ).eval()
    value = torch.randn(1, 16, 32, 48)
    torch.testing.assert_close(model(value), model.forward_static(value))
