"""生成仅用于流水线冒烟验证的合成 RAW 配对数据。

这里的程序纹理和简化噪声不代表真实手机成像分布，不能用于宣称模型质量。
它的价值是无需下载外部数据即可验证清单、训练、导出和回归测试能否闭环。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import ManifestRecord, write_manifest


def _procedural_clean_raw(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """合成含渐变、周期纹理和简单几何结构的四通道干净 RAW。"""

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x /= max(1, width - 1)
    y /= max(1, height - 1)
    phase = rng.uniform(0, 2 * np.pi)
    frequency = rng.uniform(2.0, 8.0)
    base = 0.12 + 0.34 * x + 0.24 * y
    texture = 0.08 * np.sin(2 * np.pi * frequency * x + phase)
    texture *= 0.5 + 0.5 * np.cos(2 * np.pi * (frequency / 2) * y)
    radius = np.sqrt((x - rng.uniform(0.25, 0.75)) ** 2 + (y - rng.uniform(0.25, 0.75)) ** 2)
    shape = (radius < rng.uniform(0.12, 0.3)).astype(np.float32) * rng.uniform(0.08, 0.25)
    luminance = np.clip(base + texture + shape, 0.005, 0.95)
    gains = np.asarray([1.03, 1.0, 0.99, 0.94], dtype=np.float32)[:, None, None]
    channel_texture = rng.normal(0, 0.005, size=(4, height, width)).astype(np.float32)
    return np.clip(luminance[None] * gains + channel_texture, 0.0, 1.0)


def _add_sensor_noise(
    clean: np.ndarray,
    rng: np.random.Generator,
    *,
    shot_scale: float,
    read_sigma: float,
) -> np.ndarray:
    """叠加信号相关散粒噪声、读出噪声、行列噪声和黑电平漂移。"""

    variance = shot_scale * clean + read_sigma**2
    noise = rng.normal(size=clean.shape).astype(np.float32) * np.sqrt(variance)
    row_noise = rng.normal(0, read_sigma * 0.35, size=(4, clean.shape[1], 1)).astype(np.float32)
    column_noise = rng.normal(0, read_sigma * 0.2, size=(4, 1, clean.shape[2])).astype(np.float32)
    black_drift = rng.normal(0, read_sigma * 0.15, size=(4, 1, 1)).astype(np.float32)
    return np.clip(clean + noise + row_noise + column_noise + black_drift, 0.0, 1.0)


def generate_smoke_dataset(
    output_dir: str | Path,
    *,
    samples: int = 16,
    height: int = 96,
    width: int = 96,
    seed: int = 20260726,
) -> Path:
    """生成可复现且无需外部许可的冒烟测试数据，并返回清单路径。

    数据按完整场景样本划分，噪声强度覆盖四个 ISO 桶。固定随机种子保证
    CI 的训练与数值检查可重复，但这些样本绝不能替代真实 RAW 质量评测。
    """

    if samples < 4:
        raise ValueError("at least four samples are required")
    if height % 16 or width % 16:
        raise ValueError("height and width must be multiples of 16")
    output = Path(output_dir)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[ManifestRecord] = []
    train_end = max(1, int(samples * 0.75))
    val_end = max(train_end + 1, int(samples * 0.875))
    for index in range(samples):
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "val"
        else:
            split = "test"
        # 循环覆盖四档噪声，使很小的冒烟集也能走过分桶统计路径。
        iso_index = index % 4
        iso_bucket = ("low", "medium", "high", "extreme")[iso_index]
        shot_scale = (0.0004, 0.001, 0.003, 0.008)[iso_index]
        read_sigma = (0.001, 0.002, 0.004, 0.008)[iso_index]
        clean = _procedural_clean_raw(height, width, rng)
        noisy = _add_sensor_noise(clean, rng, shot_scale=shot_scale, read_sigma=read_sigma)
        sample_id = f"smoke_{index:04d}"
        input_path = sample_dir / f"{sample_id}_input.npz"
        target_path = sample_dir / f"{sample_id}_target.npz"
        np.savez_compressed(input_path, raw=noisy)
        np.savez_compressed(target_path, raw=clean)
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                dataset_id="synthetic_smoke",
                input_path=input_path.relative_to(output).as_posix(),
                target_path=target_path.relative_to(output).as_posix(),
                split=split,
                sensor_id="smoke_sensor",
                mode="single",
                session_id=f"session_{index:04d}",
                scene_id=f"scene_{index:04d}",
                iso_bucket=iso_bucket,
                metadata={
                    "noise_sigma": min(1.0, read_sigma * 50),
                    "exposure_ratio": 1.0,
                    "wb_rg": 1.08,
                    "wb_bg": 1.12,
                    "synthetic_only": True,
                },
            )
        )
    manifest_path = output / "manifest.jsonl"
    write_manifest(records, manifest_path)
    return manifest_path
