"""基于 JSONL 清单的配对 RAW 训练数据集。

模块只接受项目定义的紧凑 NPZ 容器，并在读取时构建与部署完全一致的
16 通道输入；这样数据增强不会绕过输入契约或破坏目标配对关系。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .context import ContextBuilder, RawMetadata
from .manifest import ManifestRecord, read_manifest


class RawPairDataset(Dataset[dict[str, Any]]):
    """读取清单驱动的噪声 RAW/干净 RAW 配对数据。

    裁剪和翻转同时作用于输入、标签及所有空间条件图，保证监督像素始终对齐。
    ``split`` 在初始化阶段固定过滤，避免 DataLoader 运行中混入其他集合。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        context_builder: ContextBuilder,
        crop_size: int | None = None,
        augment: bool = False,
    ) -> None:
        """加载指定划分的记录并保存裁剪、增强和上下文构建配置。"""

        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.records = [
            record for record in read_manifest(self.manifest_path) if record.split == split
        ]
        if not self.records:
            raise ValueError(f"manifest has no records for split {split!r}")
        self.context_builder = context_builder
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        """返回当前数据划分中的配对样本数。"""

        return len(self.records)

    def _resolve(self, value: str) -> Path:
        """把清单中的相对路径解析为相对于清单目录的文件路径。"""

        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _load_raw(path: Path) -> tuple[Tensor, dict[str, Tensor]]:
        """从 NPZ 读取四通道 RAW 及可选空间条件图，禁止反序列化对象。"""

        with np.load(path, allow_pickle=False) as archive:
            if "raw" not in archive:
                raise ValueError(f"{path} does not contain a 'raw' array")
            raw = torch.from_numpy(np.asarray(archive["raw"], dtype=np.float32))
            extras = {
                name: torch.from_numpy(np.asarray(archive[name], dtype=np.float32))
                for name in ("fusion_confidence", "motion_ghost", "valid_mask")
                if name in archive
            }
        if raw.ndim != 3 or raw.shape[0] != 4:
            raise ValueError(f"{path}: raw must have shape 4×H×W, received {raw.shape}")
        return raw, extras

    def _crop(self, input_raw: Tensor, target: Tensor, extras: dict[str, Tensor]) -> tuple:
        """对输入、目标和条件图应用同一随机窗口裁剪。"""

        if self.crop_size is None:
            return input_raw, target, extras
        size = self.crop_size
        height, width = input_raw.shape[-2:]
        if height < size or width < size:
            raise ValueError(f"crop {size} is larger than sample {height}×{width}")
        top = int(torch.randint(0, height - size + 1, ()).item())
        left = int(torch.randint(0, width - size + 1, ()).item())
        crop = (..., slice(top, top + size), slice(left, left + size))
        return (
            input_raw[crop],
            target[crop],
            {
                name: value[..., top : top + size, left : left + size]
                for name, value in extras.items()
            },
        )

    def _augment(
        self, input_raw: Tensor, target: Tensor, extras: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """以独立概率执行水平/垂直翻转，并保持所有配对张量严格同步。"""

        if not self.augment:
            return input_raw, target, extras
        dimensions: list[int] = []
        if torch.rand(()).item() < 0.5:
            dimensions.append(-1)
        if torch.rand(()).item() < 0.5:
            dimensions.append(-2)
        if not dimensions:
            return input_raw, target, extras
        return (
            torch.flip(input_raw, dimensions),
            torch.flip(target, dimensions),
            {name: torch.flip(value, dimensions) for name, value in extras.items()},
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取一个样本，构建 16 通道输入并返回训练引擎需要的字段。"""

        record: ManifestRecord = self.records[index]
        input_raw, extras = self._load_raw(self._resolve(record.input_path))
        target, _ = self._load_raw(self._resolve(record.target_path))
        input_raw, target, extras = self._crop(input_raw, target, extras)
        input_raw, target, extras = self._augment(input_raw, target, extras)
        metadata_values = record.metadata
        embedding_value = metadata_values.get("camera_embedding")
        metadata = RawMetadata(
            sensor_id=record.sensor_id,
            mode=record.mode,
            noise_sigma=float(metadata_values.get("noise_sigma", 0.0)),
            exposure_ratio=float(metadata_values.get("exposure_ratio", 1.0)),
            wb_rg=float(metadata_values.get("wb_rg", 1.0)),
            wb_bg=float(metadata_values.get("wb_bg", 1.0)),
            # 清单未显式覆盖时必须传 None，让 ContextBuilder 使用传感器注册表；
            # 若擅自填零向量，会静默绕过已经通过治理门禁的相机嵌入。
            camera_embedding=(
                tuple(float(value) for value in embedding_value)  # type: ignore[arg-type]
                if embedding_value is not None
                else None
            ),
        )
        # 上下文构建放在裁剪和增强之后，可避免先生成 12 个大尺寸条件平面。
        context = self.context_builder.build(
            input_raw,
            metadata,
            fusion_confidence=extras.get("fusion_confidence"),
            motion_ghost=extras.get("motion_ghost"),
            valid_mask=extras.get("valid_mask", 1.0),
        ).squeeze(0)
        return {
            "input": context,
            "target": target,
            "sample_id": record.sample_id,
            "sensor_id": record.sensor_id,
            "mode": record.mode,
            "iso_bucket": record.iso_bucket,
        }
