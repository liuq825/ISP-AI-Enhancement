"""统一的 RAW 基线与模型质量评测入口。

评测按样本计算 PSNR，再分别汇总全局、传感器和 ISO 桶；同时记录 Manifest、
checkpoint 和模型配置哈希。公开 SIDD 结果与后续 P0/剪枝/QAT 结果必须使用本入口，
避免不同脚本在裁剪、上下文或平均方式上形成不可比较的数字。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from isp_ai_enhancement.data.context import ContextBuilder, load_context_config
from isp_ai_enhancement.data.dataset import RawPairDataset
from isp_ai_enhancement.data.governance import enforce_data_policy
from isp_ai_enhancement.data.manifest import read_manifest, validate_manifest
from isp_ai_enhancement.export import load_checkpoint_state, sha256_file
from isp_ai_enhancement.metrics import psnr_per_sample, ssim_per_sample
from isp_ai_enhancement.models.factory import build_model_from_file


@dataclass
class _MetricGroup:
    """累加一个分组的 noisy/增强 PSNR 与 packed RAW SSIM。"""

    count: int = 0
    noisy_psnr_sum: float = 0.0
    noisy_ssim_sum: float = 0.0
    enhanced_psnr_sum: float = 0.0
    enhanced_ssim_sum: float = 0.0
    has_model: bool = False

    def add(
        self,
        noisy_psnr: float,
        noisy_ssim: float,
        enhanced_psnr: float | None,
        enhanced_ssim: float | None,
    ) -> None:
        """加入一个样本；只有模型评测时才累加增强指标。"""

        self.count += 1
        self.noisy_psnr_sum += noisy_psnr
        self.noisy_ssim_sum += noisy_ssim
        if enhanced_psnr is not None and enhanced_ssim is not None:
            self.enhanced_psnr_sum += enhanced_psnr
            self.enhanced_ssim_sum += enhanced_ssim
            self.has_model = True

    def as_dict(self) -> dict[str, float | int]:
        """输出稳定 JSON 字段，并计算模型相对 noisy 输入的 PSNR 增益。"""

        if self.count <= 0:
            raise ValueError("cannot summarize an empty metric group")
        noisy = self.noisy_psnr_sum / self.count
        result: dict[str, float | int] = {
            "samples": self.count,
            "noisy_psnr_db": noisy,
            "noisy_packed_raw_ssim": self.noisy_ssim_sum / self.count,
        }
        if self.has_model:
            enhanced = self.enhanced_psnr_sum / self.count
            enhanced_ssim = self.enhanced_ssim_sum / self.count
            result["enhanced_psnr_db"] = enhanced
            result["psnr_gain_db"] = enhanced - noisy
            result["enhanced_packed_raw_ssim"] = enhanced_ssim
            result["packed_raw_ssim_gain"] = (
                enhanced_ssim - self.noisy_ssim_sum / self.count
            )
        return result


def _update_groups(
    groups: dict[str, _MetricGroup],
    keys: Iterable[str],
    noisy_psnr_values: Iterable[float],
    noisy_ssim_values: Iterable[float],
    enhanced_psnr_values: Iterable[float | None],
    enhanced_ssim_values: Iterable[float | None],
) -> None:
    """把一批逐样本指标加入指定字符串分组。"""

    for key, noisy_psnr, noisy_ssim, enhanced_psnr, enhanced_ssim in zip(
        keys,
        noisy_psnr_values,
        noisy_ssim_values,
        enhanced_psnr_values,
        enhanced_ssim_values,
        strict=True,
    ):
        groups.setdefault(str(key), _MetricGroup()).add(
            noisy_psnr,
            noisy_ssim,
            enhanced_psnr,
            enhanced_ssim,
        )


def _load_evaluation_model(
    model_config: str | Path | None,
    checkpoint: str | Path | None,
    device: torch.device,
) -> nn.Module | None:
    """成对加载模型配置与 checkpoint；两者都省略时只评估 noisy 基线。"""

    if (model_config is None) != (checkpoint is None):
        raise ValueError("model_config and checkpoint must be provided together")
    if model_config is None or checkpoint is None:
        return None
    model = build_model_from_file(model_config).to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint), strict=True)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_manifest(
    *,
    manifest: str | Path,
    split: str,
    context_config: str | Path,
    catalog: str | Path,
    purpose: str,
    model_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output: str | Path | None = None,
    device_name: str = "cpu",
    batch_size: int = 1,
    num_workers: int = 0,
) -> dict[str, Any]:
    """评估一个清单划分并返回可追溯的 JSON 兼容报告。

    先执行文件、防泄漏、许可用途和传感器注册表门禁。模型输出按残差语义与前四个
    RAW 通道相加并裁到 ``[0,1]``；没有提供模型时仍产生 noisy 基线，便于数据验收。
    """

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    manifest_path = Path(manifest)
    records = read_manifest(manifest_path)
    errors = validate_manifest(records, root=manifest_path.parent)
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"manifest validation failed:\n{formatted}")
    context = load_context_config(context_config)
    enforce_data_policy(
        records,
        catalog_path=catalog,
        purpose=purpose,
        context_config=context,
    )
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    model = _load_evaluation_model(model_config, checkpoint, device)
    dataset = RawPairDataset(
        manifest_path,
        split=split,
        context_builder=ContextBuilder(context),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    overall = _MetricGroup()
    by_sensor: dict[str, _MetricGroup] = {}
    by_iso: dict[str, _MetricGroup] = {}
    started = perf_counter()
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        valid_mask = inputs[:, 15:16]
        noisy = inputs[:, :4]
        noisy_values = psnr_per_sample(noisy, targets, valid_mask).cpu().tolist()
        noisy_ssim_values = (
            ssim_per_sample(noisy, targets, valid_mask).cpu().tolist()
        )
        if model is None:
            enhanced_values: list[float | None] = [None] * len(noisy_values)
            enhanced_ssim_values: list[float | None] = [None] * len(noisy_values)
        else:
            enhanced = torch.clamp(noisy + model(inputs), 0.0, 1.0)
            enhanced_values = [
                float(value)
                for value in psnr_per_sample(
                    enhanced, targets, valid_mask
                ).cpu().tolist()
            ]
            enhanced_ssim_values = [
                float(value)
                for value in ssim_per_sample(
                    enhanced, targets, valid_mask
                ).cpu().tolist()
            ]
        for noisy_value, noisy_ssim, enhanced_value, enhanced_ssim in zip(
            noisy_values,
            noisy_ssim_values,
            enhanced_values,
            enhanced_ssim_values,
            strict=True,
        ):
            overall.add(
                float(noisy_value),
                float(noisy_ssim),
                enhanced_value,
                enhanced_ssim,
            )
        _update_groups(
            by_sensor,
            batch["sensor_id"],
            noisy_values,
            noisy_ssim_values,
            enhanced_values,
            enhanced_ssim_values,
        )
        _update_groups(
            by_iso,
            batch["iso_bucket"],
            noisy_values,
            noisy_ssim_values,
            enhanced_values,
            enhanced_ssim_values,
        )

    report: dict[str, Any] = {
        "format_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split": split,
        "purpose": purpose,
        "device": str(device),
        "elapsed_seconds": perf_counter() - started,
        "overall": overall.as_dict(),
        "by_sensor": {
            key: value.as_dict() for key, value in sorted(by_sensor.items())
        },
        "by_iso_bucket": {
            key: value.as_dict() for key, value in sorted(by_iso.items())
        },
        "model": None,
    }
    if model_config is not None and checkpoint is not None:
        report["model"] = {
            "config": str(model_config),
            "config_sha256": sha256_file(model_config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
