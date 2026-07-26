"""验证清单数据集读取与相机注册表上下文的真实联动。"""

from pathlib import Path

import numpy as np
import torch

from isp_ai_enhancement.data.context import ContextBuilder, ContextConfig
from isp_ai_enhancement.data.dataset import RawPairDataset
from isp_ai_enhancement.data.manifest import ManifestRecord, write_manifest


def test_dataset_uses_registered_camera_embedding_when_manifest_has_no_override(
    tmp_path: Path,
) -> None:
    """清单未内嵌相机向量时，应使用注册表而不是静默回退零向量。"""

    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    raw = np.full((4, 8, 8), 0.25, dtype=np.float32)
    np.savez_compressed(sample_dir / "input.npz", raw=raw)
    np.savez_compressed(sample_dir / "target.npz", raw=raw)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            ManifestRecord(
                sample_id="registered_sensor",
                dataset_id="test",
                input_path="samples/input.npz",
                target_path="samples/target.npz",
                split="train",
                sensor_id="sensor",
                mode="single",
                session_id="session",
                scene_id="scene",
                iso_bucket="low",
                metadata={},
            )
        ],
        manifest,
    )
    embedding = (0.1, -0.2, 0.3, -0.4)
    dataset = RawPairDataset(
        manifest,
        split="train",
        context_builder=ContextBuilder(
            ContextConfig(camera_embeddings={"sensor": embedding})
        ),
    )
    context = dataset[0]["input"]
    for channel, expected in zip(range(8, 12), embedding, strict=True):
        torch.testing.assert_close(context[channel], torch.full((8, 8), expected))


def test_validation_crop_is_deterministic_center_window(tmp_path: Path) -> None:
    """关闭增强的验证集应固定中心裁剪，且不消耗全局随机数。"""

    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    raw = np.arange(4 * 8 * 10, dtype=np.float32).reshape(4, 8, 10) / 1000.0
    np.savez_compressed(sample_dir / "input.npz", raw=raw)
    np.savez_compressed(sample_dir / "target.npz", raw=raw)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            ManifestRecord(
                sample_id="center_crop",
                dataset_id="test",
                input_path="samples/input.npz",
                target_path="samples/target.npz",
                split="val",
                sensor_id="sensor",
                mode="single",
                session_id="session",
                scene_id="scene",
                iso_bucket="low",
                metadata={},
            )
        ],
        manifest,
    )
    dataset = RawPairDataset(
        manifest,
        split="val",
        context_builder=ContextBuilder(
            ContextConfig(camera_embeddings={"sensor": (0.0, 0.0, 0.0, 0.0)})
        ),
        crop_size=4,
        augment=False,
    )

    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()
    first = dataset[0]["target"]
    second = dataset[0]["target"]
    torch.testing.assert_close(first, torch.from_numpy(raw[:, 2:6, 3:7]))
    torch.testing.assert_close(second, first)
    torch.testing.assert_close(torch.get_rng_state(), state_before)
