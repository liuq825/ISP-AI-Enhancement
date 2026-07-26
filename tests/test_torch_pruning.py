"""验证 Torch-Pruning 对 NAFNet 成对门控通道的物理裁剪行为。"""

from importlib.metadata import version

import torch

from isp_ai_enhancement.models.nafnet import ExpansionSpec, NAFNetRaw
from isp_ai_enhancement.pruning.torch_pruning_adapter import (
    torch_pruning_physical_prune,
)


def test_torch_pruning_rebuilds_simple_gate_dependencies() -> None:
    """裁剪一个门控宽度后，SCA、投影卷积和前向 shape 必须全部同步。"""

    torch.manual_seed(17)
    source = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    ).eval()
    values = source.expansion_spec.as_dict()
    values["enc1"] = [3]
    target_spec = ExpansionSpec.from_mapping(values)
    target, report, backend = torch_pruning_physical_prune(source, target_spec)
    first = target.encoders[0][0]

    assert first.conv1.out_channels == 6
    assert first.conv2.groups == 6
    assert first.sca_conv.in_channels == 3
    assert first.conv3.in_channels == 3
    assert first.conv4.out_channels == 6
    assert first.conv5.in_channels == 3
    assert report.target_parameters < report.source_parameters
    assert backend.dependency_groups == 2
    assert backend.dependency_operations >= backend.dependency_groups
    assert backend.pruned_gate_units == 2
    assert target(torch.randn(1, 16, 16, 16)).shape == (1, 4, 16, 16)


def test_torch_pruning_noop_keeps_numerical_output() -> None:
    """目标宽度不变时不应创建依赖组，输出必须逐元素保持一致。"""

    torch.manual_seed(19)
    source = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    ).eval()
    value = torch.randn(1, 16, 16, 16)
    target, report, backend = torch_pruning_physical_prune(
        source, source.expansion_spec
    )
    assert report.pruning_ratio == 0
    assert backend.dependency_groups == 0
    assert backend.dependency_operations == 0
    assert backend.backend_version == version("torch-pruning")
    torch.testing.assert_close(source(value), target(value))
