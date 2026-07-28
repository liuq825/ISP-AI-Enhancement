"""验证 feature + attention 联合蒸馏的数值、梯度和恢复语义。"""

import pytest
import torch

from isp_ai_enhancement.distillation import (
    FeatureAttentionDistiller,
    spatial_attention_map,
)


def _feature_pyramid(width: int, *, requires_grad: bool) -> dict[str, torch.Tensor]:
    """生成通道倍率和空间倍率均符合四级 NAFNet 的最小特征金字塔。"""

    return {
        "enc2": torch.randn(2, width * 2, 8, 8, requires_grad=requires_grad),
        "enc3": torch.randn(2, width * 4, 4, 4, requires_grad=requires_grad),
        "middle": torch.randn(2, width * 16, 1, 1, requires_grad=requires_grad),
    }


def test_feature_attention_distillation_backpropagates_only_to_student() -> None:
    """联合损失应更新 Student 和投影层，但不得向固定 Teacher 传播梯度。"""

    student = _feature_pyramid(2, requires_grad=True)
    teacher = _feature_pyramid(4, requires_grad=True)
    distiller = FeatureAttentionDistiller(
        student_width=2,
        teacher_width=4,
        feature_keys=("enc2", "middle"),
        attention_keys=("enc3", "middle"),
    )

    terms = distiller(student, teacher)
    assert set(terms) == {"feature", "attention"}
    assert terms["feature"].ndim == terms["attention"].ndim == 0
    assert torch.isfinite(terms["feature"])
    assert torch.isfinite(terms["attention"])
    (0.15 * terms["feature"] + 0.10 * terms["attention"]).backward()

    assert all(value.grad is not None for value in student.values())
    assert all(value.grad is None for value in teacher.values())
    assert all(parameter.grad is not None for parameter in distiller.parameters())


def test_attention_map_is_finite_and_per_sample_normalized() -> None:
    """非零特征 attention 的 L2 范数应为 1，全零特征也不能产生 NaN。"""

    feature = torch.randn(3, 5, 4, 6)
    attention = spatial_attention_map(feature)
    norms = attention.flatten(start_dim=1).norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms))

    zero_attention = spatial_attention_map(torch.zeros(2, 3, 4, 4))
    assert torch.count_nonzero(zero_attention) == 0
    assert torch.isfinite(zero_attention).all()


def test_attention_key_change_is_rejected_by_strict_checkpoint_restore() -> None:
    """attention 层集合虽无参数，改变后仍必须触发严格状态恢复失败。"""

    source = FeatureAttentionDistiller(
        2,
        4,
        feature_keys=("enc2",),
        attention_keys=("enc3",),
    )
    changed = FeatureAttentionDistiller(
        2,
        4,
        feature_keys=("enc2",),
        attention_keys=("middle",),
    )

    with pytest.raises(RuntimeError, match="state_dict"):
        changed.load_state_dict(source.state_dict(), strict=True)
