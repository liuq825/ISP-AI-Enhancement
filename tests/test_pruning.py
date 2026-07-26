import torch

from isp_ai_enhancement.models.nafnet import NAFNetRaw, reference_pruned_spec
from isp_ai_enhancement.pruning.physical import physical_prune


def test_physical_pruning_rebuilds_smaller_graph() -> None:
    torch.manual_seed(7)
    source = NAFNetRaw()
    target, report = physical_prune(source, reference_pruned_spec())
    assert target.parameter_count() == report.target_parameters
    assert 0.14 <= report.pruning_ratio <= 0.16
    assert all(parameter.numel() for parameter in target.parameters())


def test_noop_pruning_preserves_output() -> None:
    torch.manual_seed(9)
    source = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    ).eval()
    target, report = physical_prune(source, source.expansion_spec)
    value = torch.randn(1, 16, 16, 16)
    assert report.pruning_ratio == 0
    torch.testing.assert_close(source(value), target(value))
