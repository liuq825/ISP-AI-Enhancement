"""Student 到 Teacher 的多尺度特征投影与蒸馏损失。"""

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


class FeatureDistiller(nn.Module):
    """仅训练期使用 1×1 卷积对齐 Student/Teacher 通道并计算特征损失。"""

    def __init__(
        self,
        student_width: int,
        teacher_width: int,
        keys: Sequence[str] = ("enc2", "enc4", "middle", "dec2"),
    ) -> None:
        """校验特征层名称，并按各 stage 的倍率创建独立投影层。"""

        super().__init__()
        unknown = sorted(set(keys) - _STAGE_MULTIPLIERS.keys())
        if unknown:
            raise ValueError(f"unknown distillation feature keys: {unknown}")
        if not keys:
            raise ValueError("at least one distillation feature key is required")
        self.keys = tuple(keys)
        self.projections = nn.ModuleDict(
            {
                key: nn.Conv2d(
                    student_width * _STAGE_MULTIPLIERS[key],
                    teacher_width * _STAGE_MULTIPLIERS[key],
                    1,
                )
                for key in self.keys
            }
        )

    def forward(
        self,
        student_features: Mapping[str, Tensor],
        teacher_features: Mapping[str, Tensor],
    ) -> Tensor:
        """逐层投影 Student 特征后与停止梯度的 Teacher 特征计算 Smooth-L1。"""

        losses: list[Tensor] = []
        for key in self.keys:
            if key not in student_features or key not in teacher_features:
                raise ValueError(f"missing feature {key!r} for distillation")
            projected = self.projections[key](student_features[key])
            teacher = teacher_features[key].detach()
            if projected.shape != teacher.shape:
                raise ValueError(
                    f"feature {key} shape mismatch: "
                    f"{tuple(projected.shape)} != {tuple(teacher.shape)}"
                )
            losses.append(F.smooth_l1_loss(projected, teacher))
        return sum(losses) / len(losses)
