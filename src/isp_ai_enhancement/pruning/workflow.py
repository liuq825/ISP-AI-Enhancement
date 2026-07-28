"""从已训练 FP32 checkpoint 生成可追溯的物理剪枝 checkpoint。

工作流严格区分源模型配置和目标扩展规格，执行选定后端后再用目标配置重建模型并
严格加载权重，避免只得到内存中可运行、却无法由配置复现的临时计算图。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from isp_ai_enhancement.export import load_checkpoint_state, sha256_file
from isp_ai_enhancement.models.factory import build_model_from_file
from isp_ai_enhancement.models.nafnet import NAFNetRaw

from .physical import PruningReport, physical_prune, stage_hidden_retention
from .torch_pruning_adapter import (
    TorchPruningReport,
    torch_pruning_physical_prune,
)


def _validate_compatible_backbones(source: NAFNetRaw, target: NAFNetRaw) -> None:
    """要求剪枝前后只改变块内扩展宽度，不改变网络公共主干。"""

    fields = (
        "input_channels",
        "output_channels",
        "width",
        "encoder_blocks",
        "middle_blocks_count",
        "decoder_blocks",
    )
    mismatches = [
        name for name in fields if getattr(source, name) != getattr(target, name)
    ]
    if mismatches:
        raise ValueError(
            "structured pruning target changes unsupported backbone fields: "
            + ", ".join(mismatches)
        )


def _backend_metadata(
    backend: str,
    report: TorchPruningReport | None,
) -> dict[str, Any]:
    """把剪枝后端信息转换为稳定的 JSON 兼容元数据。"""

    if report is None:
        return {"name": backend}
    return {
        "name": backend,
        "version": report.backend_version,
        "dependency_groups": report.dependency_groups,
        "dependency_operations": report.dependency_operations,
        "pruned_gate_units": report.pruned_gate_units,
    }


def prune_checkpoint(
    *,
    source_config: str | Path,
    source_checkpoint: str | Path,
    target_config: str | Path,
    output: str | Path,
    backend: str = "torch-pruning",
) -> tuple[Path, dict[str, Any]]:
    """物理剪枝已训练权重，验证目标配置可重建性并原子保存产物。

    ``manual`` 与 ``torch-pruning`` 使用相同的重要性分数和成对 SimpleGate 索引。
    输出仍是 FP32 checkpoint，必须在真实验证集微调并通过画质 Gate 后才能进入 QAT。
    """

    if backend not in {"manual", "torch-pruning"}:
        raise ValueError("backend must be manual or torch-pruning")
    source_config_path = Path(source_config)
    source_checkpoint_path = Path(source_checkpoint)
    target_config_path = Path(target_config)
    source = build_model_from_file(source_config_path)
    source.load_state_dict(
        load_checkpoint_state(source_checkpoint_path),
        strict=True,
    )
    source.eval()
    configured_target = build_model_from_file(target_config_path)
    _validate_compatible_backbones(source, configured_target)

    backend_report: TorchPruningReport | None = None
    structural_report: PruningReport
    if backend == "torch-pruning":
        pruned, structural_report, backend_report = torch_pruning_physical_prune(
            source,
            configured_target.expansion_spec,
        )
    else:
        pruned, structural_report = physical_prune(
            source,
            configured_target.expansion_spec,
        )
    pruned.eval()

    # 严格加载到“从目标 YAML 新建”的模型，证明产物不依赖内存对象上的临时属性修改。
    rebuilt = build_model_from_file(target_config_path)
    rebuilt.load_state_dict(pruned.state_dict(), strict=True)
    rebuilt.eval()
    with torch.inference_mode():
        sample = torch.rand(1, source.input_channels, 16, 16)
        torch.testing.assert_close(
            pruned(sample),
            rebuilt(sample),
            rtol=0,
            atol=0,
        )

    metadata: dict[str, Any] = {
        "format_version": 1,
        "artifact_type": "structured_pruned_fp32_checkpoint",
        "source_config": str(source_config_path),
        "source_config_sha256": sha256_file(source_config_path),
        "source_checkpoint": str(source_checkpoint_path),
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "target_config": str(target_config_path),
        "target_config_sha256": sha256_file(target_config_path),
        "source_parameters": structural_report.source_parameters,
        "target_parameters": structural_report.target_parameters,
        "physical_pruning_ratio": structural_report.pruning_ratio,
        "stage_hidden_retention": stage_hidden_retention(
            source.expansion_spec,
            configured_target.expansion_spec,
        ),
        "backend": _backend_metadata(backend, backend_report),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    torch.save(
        {
            "format_version": 1,
            "model_state": rebuilt.state_dict(),
            "pruning": metadata,
        },
        temporary,
    )
    temporary.replace(output_path)
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    manifest_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, metadata
