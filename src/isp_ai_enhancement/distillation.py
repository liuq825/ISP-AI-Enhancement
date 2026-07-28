"""Student 到 Teacher 的多尺度特征与空间 attention 联合蒸馏。

特征蒸馏通过训练期 1×1 投影对齐不同宽度的 Student/Teacher；attention 蒸馏先把
每层特征沿通道聚合为空间能量图，再做逐样本 L2 归一化，因此无需强行对齐通道数。
这些适配器只参与训练，不进入导出的 Student 部署图。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor, nn
from torch.nn import functional as F

_STAGE_MULTIPLIERS = {
    "enc1": 1,
    "enc2": 2,
    "enc3": 4,
    "enc4": 8,
    "middle": 16,
    "dec1": 8,
    "dec2": 4,
    "dec3": 2,
    "dec4": 1,
}


def spatial_attention_map(feature: Tensor, epsilon: float = 1e-6) -> Tensor:
    """把 ``N×C×H×W`` 特征转换为逐样本归一化的空间 attention 图。

    attention 使用通道均方能量，既保留纹理、边缘和噪声残差集中区域，又不依赖
    Teacher/Student 通道数相同。展平后的 L2 归一化去除绝对激活尺度，避免宽 Teacher
    仅因通道更多而主导损失；``epsilon`` 防止全零特征产生 NaN。
    """

    if feature.ndim != 4 or min(feature.shape) <= 0:
        raise ValueError(
            f"attention feature must be non-empty N×C×H×W, got {tuple(feature.shape)}"
        )
    energy = feature.square().mean(dim=1, keepdim=True)
    flattened = energy.flatten(start_dim=1)
    normalized = flattened / flattened.norm(dim=1, keepdim=True).clamp_min(epsilon)
    return normalized.view_as(energy)


class FeatureAttentionDistiller(nn.Module):
    """联合计算投影特征损失和归一化空间 attention 迁移损失。"""

    def __init__(
        self,
        student_width: int,
        teacher_width: int,
        feature_keys: Sequence[str] = ("enc2", "enc3", "enc4", "middle", "dec2"),
        attention_keys: Sequence[str] = ("enc3", "enc4", "middle", "dec1", "dec2"),
    ) -> None:
        """校验两组层名，并仅为需要通道对齐的 feature 分支创建投影层。

        每个 attention key 还注册一个零字节语义标记 buffer。它不参与计算，但会写入
        checkpoint；恢复时若 attention 层集合改变，``strict=True`` 会因状态键不一致
        而拒绝，避免无参数配置变化绕过断点恢复一致性检查。
        """

        super().__init__()
        unknown = sorted(
            (set(feature_keys) | set(attention_keys)) - _STAGE_MULTIPLIERS.keys()
        )
        if unknown:
            raise ValueError(f"unknown distillation feature keys: {unknown}")
        if not feature_keys:
            raise ValueError("at least one feature distillation key is required")
        if not attention_keys:
            raise ValueError("at least one attention distillation key is required")
        if len(set(feature_keys)) != len(feature_keys):
            raise ValueError("feature distillation keys must not contain duplicates")
        if len(set(attention_keys)) != len(attention_keys):
            raise ValueError("attention distillation keys must not contain duplicates")
        self.feature_keys = tuple(feature_keys)
        self.attention_keys = tuple(attention_keys)
        self.projections = nn.ModuleDict(
            {
                key: nn.Conv2d(
                    student_width * _STAGE_MULTIPLIERS[key],
                    teacher_width * _STAGE_MULTIPLIERS[key],
                    1,
                )
                for key in self.feature_keys
            }
        )
        for key in self.attention_keys:
            self.register_buffer(
                f"_attention_key_{key}",
                Tensor(),
                persistent=True,
            )

    def forward(
        self,
        student_features: Mapping[str, Tensor],
        teacher_features: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """返回未加权的 feature 与 attention 两项损失。

        Teacher 张量在本函数内再次 ``detach``，形成防误用边界；总权重由训练配置统一
        管理并记录到 checkpoint，便于后续做逐项消融而不修改蒸馏模块结构。
        """

        feature_losses: list[Tensor] = []
        for key in self.feature_keys:
            if key not in student_features or key not in teacher_features:
                raise ValueError(f"missing feature {key!r} for distillation")
            projected = self.projections[key](student_features[key])
            teacher = teacher_features[key].detach()
            if projected.shape != teacher.shape:
                raise ValueError(
                    f"feature {key} shape mismatch: "
                    f"{tuple(projected.shape)} != {tuple(teacher.shape)}"
                )
            feature_losses.append(F.smooth_l1_loss(projected, teacher))

        attention_losses: list[Tensor] = []
        for key in self.attention_keys:
            if key not in student_features or key not in teacher_features:
                raise ValueError(f"missing attention feature {key!r} for distillation")
            student_attention = spatial_attention_map(student_features[key])
            teacher_attention = spatial_attention_map(teacher_features[key].detach())
            if student_attention.shape != teacher_attention.shape:
                raise ValueError(
                    f"attention {key} shape mismatch: "
                    f"{tuple(student_attention.shape)} != "
                    f"{tuple(teacher_attention.shape)}"
                )
            attention_losses.append(F.mse_loss(student_attention, teacher_attention))

        return {
            "feature": sum(feature_losses) / len(feature_losses),
            "attention": sum(attention_losses) / len(attention_losses),
        }


# 兼容早期内部导入名；新代码和文档统一使用 FeatureAttentionDistiller。
FeatureDistiller = FeatureAttentionDistiller
