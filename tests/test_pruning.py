"""验证手工结构化剪枝基线的参数缩减与数值保持行为。"""

import torch

from isp_ai_enhancement.models.nafnet import (
    NAFNetRaw,
    structure_aware_pruned_spec,
)
from isp_ai_enhancement.pruning.physical import physical_prune, stage_hidden_retention


def test_physical_pruning_rebuilds_smaller_graph() -> None:
    """结构感知规格应真正移除约 15% 参数，而不是只写零掩码。"""

    torch.manual_seed(7)
    source = NAFNetRaw()
    spec = structure_aware_pruned_spec()
    target, report = physical_prune(source, spec)
    assert target.parameter_count() == report.target_parameters
    assert report.source_parameters == 14_586_340
    assert report.target_parameters == 12_405_108
    assert 0.149 <= report.pruning_ratio <= 0.151
    # 浅层完整保留；深层 stage 首尾块比内部块宽，证明不是统一比例模板。
    assert spec.enc1 == source.expansion_spec.enc1
    assert spec.enc2 == source.expansion_spec.enc2
    assert spec.enc3[0] == spec.enc3[-1] > min(spec.enc3[1:-1])
    assert spec.enc4[0] == spec.enc4[-1] > spec.enc4[1] > min(spec.enc4[2:-2])
    retention = stage_hidden_retention(source.expansion_spec, spec)
    assert retention["enc1"] == retention["enc2"] == 1.0
    assert retention["enc4"] == 0.875
    assert retention["middle"] == 0.8203125
    assert retention["dec3"] == retention["dec4"] == 1.0
    assert all(parameter.numel() for parameter in target.parameters())


def test_noop_pruning_preserves_output() -> None:
    """目标宽度不变时，重建模型输出必须与源模型一致。"""

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
