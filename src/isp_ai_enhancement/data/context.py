"""RAW 输入契约与元数据上下文构建。

部署接口固定接收 16 通道张量：前 4 通道是按 ``[R, Gr, Gb, B]`` 排列的
Bayer 数据，后 12 通道是噪声、曝光、融合置信度、相机嵌入等条件信息。
训练、导出和端侧推理必须共用本模块，避免通道顺序在不同阶段悄然漂移。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from isp_ai_enhancement.config import load_yaml

_CFA_POSITIONS: dict[str, tuple[tuple[int, int], ...]] = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (1, 1), (0, 0), (0, 1)),
    "BGGR": ((1, 1), (1, 0), (0, 1), (0, 0)),
}


def canonical_pack_bayer(raw: Tensor, cfa_pattern: str = "RGGB") -> Tensor:
    """把二维 Bayer 马赛克按 CFA 规则打包成统一的 ``[R, Gr, Gb, B]`` 顺序。

    输入可为单张 ``H×W`` 或批量 ``N×H×W``；输出空间尺寸减半、通道数为 4。
    统一顺序可使来自不同手机传感器的样本共享同一个网络输入定义。
    """
    if raw.ndim not in (2, 3):
        raise ValueError("raw must be H×W or N×H×W")
    if raw.shape[-2] % 2 or raw.shape[-1] % 2:
        raise ValueError("Bayer dimensions must be even")
    pattern = cfa_pattern.upper()
    if pattern not in _CFA_POSITIONS:
        raise ValueError(f"unsupported CFA pattern: {cfa_pattern}")
    channels = [raw[..., row::2, column::2] for row, column in _CFA_POSITIONS[pattern]]
    channel_axis = 0 if raw.ndim == 2 else 1
    return torch.stack(channels, dim=channel_axis)


@dataclass(frozen=True)
class RawMetadata:
    """构建条件通道所需的单个 RAW 样本元数据。

    数值在 :class:`ContextBuilder` 中归一化。``camera_embedding`` 允许数据记录
    显式覆盖注册表，但正式数据应优先使用版本化的相机嵌入注册表。
    """

    sensor_id: str
    mode: str = "single"
    noise_sigma: float = 0.0
    exposure_ratio: float = 1.0
    wb_rg: float = 1.0
    wb_bg: float = 1.0
    camera_embedding: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ContextConfig:
    """16 通道上下文编码的版本化配置。

    相机嵌入固定占 4 个通道；模式编码、曝光和白平衡均被压到 ``[0, 1]``，
    使训练与 INT8 量化时的动态范围可控。
    """

    max_abs_ev: float = 8.0
    wb_ratio_min: float = 0.25
    wb_ratio_max: float = 4.0
    camera_embeddings: Mapping[str, tuple[float, float, float, float]] = field(default_factory=dict)
    mode_codes: Mapping[str, float] = field(
        default_factory=lambda: {"single": 0.0, "hdr": 0.5, "mfnr": 1.0}
    )

    def __post_init__(self) -> None:
        """在配置创建时拒绝无效范围，避免错误延迟到训练阶段才暴露。"""
        if self.max_abs_ev <= 0:
            raise ValueError("max_abs_ev must be positive")
        if self.wb_ratio_min <= 0 or self.wb_ratio_max <= self.wb_ratio_min:
            raise ValueError("white-balance range must be positive and increasing")
        for sensor_id, embedding in self.camera_embeddings.items():
            if not sensor_id:
                raise ValueError("camera embedding sensor_id must not be empty")
            if len(embedding) != 4 or any(abs(float(value)) > 1.0 for value in embedding):
                raise ValueError(
                    f"camera embedding for {sensor_id!r} must contain four values in [-1, 1]"
                )
        if not self.mode_codes:
            raise ValueError("mode_codes must not be empty")
        for mode, code in self.mode_codes.items():
            if not mode or not 0.0 <= float(code) <= 1.0:
                raise ValueError(f"mode code for {mode!r} must be in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContextConfig:
        """从 YAML 映射创建严格配置，并拒绝拼写错误的未知字段。"""
        allowed = {
            "max_abs_ev",
            "wb_ratio_min",
            "wb_ratio_max",
            "camera_embeddings",
            "mode_codes",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown context configuration keys: {unknown}")
        embeddings_value = value.get("camera_embeddings", {})
        mode_codes_value = value.get(
            "mode_codes", {"single": 0.0, "hdr": 0.5, "mfnr": 1.0}
        )
        if not isinstance(embeddings_value, Mapping):
            raise ValueError("camera_embeddings must be a mapping")
        if not isinstance(mode_codes_value, Mapping):
            raise ValueError("mode_codes must be a mapping")
        embeddings = {
            str(sensor_id): tuple(float(item) for item in embedding)
            for sensor_id, embedding in embeddings_value.items()
        }
        mode_codes = {str(mode).lower(): float(code) for mode, code in mode_codes_value.items()}
        return cls(
            max_abs_ev=float(value.get("max_abs_ev", 8.0)),
            wb_ratio_min=float(value.get("wb_ratio_min", 0.25)),
            wb_ratio_max=float(value.get("wb_ratio_max", 4.0)),
            camera_embeddings=embeddings,
            mode_codes=mode_codes,
        )


def load_context_config(path: str | Path) -> ContextConfig:
    """从 YAML 文件加载上下文配置，兼容带或不带 ``context`` 顶层键的写法。"""

    value = load_yaml(path)
    context_value = value.get("context", value)
    if not isinstance(context_value, Mapping):
        raise ValueError(f"{path}: 'context' must be a mapping")
    return ContextConfig.from_mapping(context_value)


class ContextBuilder:
    """构建冻结的 16 通道部署输入契约。

    通道顺序为：4 路 RAW、噪声、曝光、融合置信度、运动鬼影、4 路相机嵌入、
    两路白平衡、拍摄模式和有效区域掩码。该顺序是模型 ABI，修改时必须同步
    更新输入契约文档、训练数据、ONNX 导出和端侧实现。
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        """保存上下文配置；未传入时使用不含相机注册项的安全默认值。"""

        self.config = config or ContextConfig()

    @staticmethod
    def _plane(value: float, reference: Tensor) -> Tensor:
        """把一个标量广播为空间条件平面，并继承参考张量的设备与数据类型。"""

        return torch.full_like(reference[:, :1], float(value))

    @staticmethod
    def _map_or_plane(value: Tensor | float, reference: Tensor, name: str) -> Tensor:
        """把条件值规范为 ``N×1×H×W``，同时校验空间条件图的精确形状。"""

        if isinstance(value, Tensor):
            result = value
            if result.ndim == 2:
                result = result.unsqueeze(0).unsqueeze(0)
            elif result.ndim == 3:
                result = result.unsqueeze(1)
            if result.shape != reference[:, :1].shape:
                raise ValueError(
                    f"{name} must have shape {tuple(reference[:, :1].shape)}, "
                    f"received {tuple(result.shape)}"
                )
            return result.to(device=reference.device, dtype=reference.dtype)
        return ContextBuilder._plane(float(value), reference)

    def _normalize_exposure(self, ratio: float) -> float:
        """把曝光比转换到 EV 空间并线性归一化到 ``[0, 1]``。"""

        if ratio <= 0:
            raise ValueError("exposure_ratio must be positive")
        ev = torch.log2(torch.tensor(float(ratio))).item()
        scaled = (ev + self.config.max_abs_ev) / (2 * self.config.max_abs_ev)
        return min(1.0, max(0.0, scaled))

    def _normalize_wb(self, ratio: float) -> float:
        """在对数域归一化白平衡增益，降低极端增益对动态范围的影响。"""

        if ratio <= 0:
            raise ValueError("white-balance ratios must be positive")
        low = torch.log2(torch.tensor(self.config.wb_ratio_min)).item()
        high = torch.log2(torch.tensor(self.config.wb_ratio_max)).item()
        value = torch.log2(torch.tensor(float(ratio))).item()
        return min(1.0, max(0.0, (value - low) / (high - low)))

    def build(
        self,
        packed_raw: Tensor,
        metadata: RawMetadata,
        *,
        fusion_confidence: Tensor | float | None = None,
        motion_ghost: Tensor | float | None = None,
        valid_mask: Tensor | float = 1.0,
    ) -> Tensor:
        """拼接 RAW 与条件信息并返回 ``N×16×H×W`` 输入张量。

        本方法执行有限值、幅值、相机注册和模式校验。单帧模式的融合置信度
        默认设为 1，多帧模式未提供置信图时默认设为 0，防止虚构融合质量。
        """

        if packed_raw.ndim == 3:
            packed_raw = packed_raw.unsqueeze(0)
        if packed_raw.ndim != 4 or packed_raw.shape[1] != 4:
            raise ValueError("packed_raw must be N×4×H×W or 4×H×W")
        raw = packed_raw.to(dtype=torch.float32)
        if not torch.isfinite(raw).all():
            raise ValueError("packed_raw contains NaN or infinity")
        if raw.min().item() < 0 or raw.max().item() > 1:
            raise ValueError("packed_raw must be black/white-level normalized to [0, 1]")

        mode = metadata.mode.lower()
        if mode not in self.config.mode_codes:
            raise ValueError(f"unknown mode: {metadata.mode}")
        if fusion_confidence is None:
            fusion_confidence = 1.0 if mode == "single" else 0.0
        if motion_ghost is None:
            motion_ghost = 0.0

        embedding = metadata.camera_embedding or self.config.camera_embeddings.get(
            metadata.sensor_id
        )
        if embedding is None:
            raise ValueError(f"missing camera embedding for sensor {metadata.sensor_id!r}")
        if len(embedding) != 4 or any(abs(value) > 1.0 for value in embedding):
            raise ValueError("camera embedding must contain four values in [-1, 1]")

        # 这里的列表顺序就是对外部署 ABI，禁止仅在训练侧随意调整。
        channels = [
            raw,
            self._plane(min(1.0, max(0.0, metadata.noise_sigma)), raw),
            self._plane(self._normalize_exposure(metadata.exposure_ratio), raw),
            self._map_or_plane(fusion_confidence, raw, "fusion_confidence").clamp(0, 1),
            self._map_or_plane(motion_ghost, raw, "motion_ghost").clamp(0, 1),
            *[self._plane(value, raw) for value in embedding],
            self._plane(self._normalize_wb(metadata.wb_rg), raw),
            self._plane(self._normalize_wb(metadata.wb_bg), raw),
            self._plane(float(self.config.mode_codes[mode]), raw),
            self._map_or_plane(valid_mask, raw, "valid_mask").clamp(0, 1),
        ]
        result = torch.cat(channels, dim=1)
        if result.shape[1] != 16:
            raise AssertionError(f"context contract produced {result.shape[1]} channels")
        return result
