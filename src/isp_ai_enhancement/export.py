"""把训练 checkpoint 导出为静态 ONNX，并执行 Checker、ORT 数值对照与哈希落盘。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from isp_ai_enhancement.config import load_yaml
from isp_ai_enhancement.models.factory import build_model_from_file
from isp_ai_enhancement.onnx_audit import audit_onnx
from isp_ai_enhancement.quantization.fake_quant import (
    SymmetricFakeQuant,
    prepare_qat,
    set_observer_enabled,
)

_INPUT_CHANNEL_SEMANTICS = [
    "raw_r",
    "raw_gr",
    "raw_gb",
    "raw_b",
    "noise_sigma",
    "exposure_log2_normalized",
    "fusion_confidence",
    "motion_ghost",
    "camera_embedding_0",
    "camera_embedding_1",
    "camera_embedding_2",
    "camera_embedding_3",
    "white_balance_rg_log2_normalized",
    "white_balance_bg_log2_normalized",
    "capture_mode",
    "valid_mask",
]


class _StaticExportWrapper(nn.Module):
    """强制导出器走无动态 Pad 的 ``forward_static`` 路径。"""

    def __init__(self, model: nn.Module) -> None:
        """保存待导出的 NAFNet 模型，不改变其参数命名。"""

        super().__init__()
        self.model = model

    def forward(self, value: Tensor) -> Tensor:
        """返回固定输入尺寸对应的四通道 RAW 残差。"""

        return self.model.forward_static(value)


def sha256_file(path: str | Path) -> str:
    """以 1 MiB 分块计算文件 SHA256，避免把大型 ONNX 一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    """兼容完整训练 checkpoint 和裸 state_dict，并统一映射到 CPU。"""

    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict) and "model_state" in value:
        return value["model_state"]
    if isinstance(value, dict):
        return value
    raise ValueError("checkpoint must contain a state dictionary")


def export_onnx(
    *,
    model_config: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    export_config: str | Path | None = None,
    qat_config: str | Path | None = None,
) -> Path:
    """导出静态 ONNX、验证 PyTorch/ORT 一致性并生成部署清单。

    输入 H/W 必须是 16 的倍数，确保图中不残留 Shape/Pad/Slice 动态子图。
    只有 Checker 和数值对照通过后才写 manifest；目标 NPU 支持仍需 DDK 实测。
    """

    model = build_model_from_file(model_config)
    qat_report = None
    qat_config_path = Path(qat_config) if qat_config is not None else None
    if qat_config_path is not None:
        qat_settings = load_yaml(qat_config_path).get("qat")
        if not isinstance(qat_settings, dict):
            raise ValueError("qat_config must contain a 'qat' mapping")
        activation_bits = int(qat_settings.get("activation_bits", 8))
        weight_bits = int(qat_settings.get("weight_bits", 8))
        if activation_bits != 8 or weight_bits != 8:
            raise ValueError("ONNX Q/DQ export currently requires 8-bit activations and weights")
        exclude_modules = qat_settings.get("exclude_modules", ("intro", "ending"))
        if not isinstance(exclude_modules, (list, tuple)) or not all(
            isinstance(value, str) for value in exclude_modules
        ):
            raise ValueError("QAT exclude_modules must be a list of module names")
        # 必须先按训练时的相同规则重建 QAT 模块树，再 strict 加载观察器 buffer；
        # 直接把 QAT checkpoint 加到普通 Conv2d 图会丢失量化尺度或键不匹配。
        qat_report = prepare_qat(
            model,
            activation_bits=activation_bits,
            weight_bits=weight_bits,
            observer_momentum=float(qat_settings.get("observer_momentum", 0.95)),
            exclude_modules=tuple(exclude_modules),
        )
    model.load_state_dict(load_checkpoint_state(checkpoint), strict=True)
    model.eval()
    if qat_report is not None:
        uninitialized = [
            name
            for name, module in model.named_modules()
            if isinstance(module, SymmetricFakeQuant)
            and not bool(module.observer_initialized.item())
        ]
        if uninitialized:
            raise ValueError(
                "QAT checkpoint contains uninitialized observers: "
                + ", ".join(uninitialized[:10])
            )
        set_observer_enabled(model, False)
    settings = load_yaml(export_config).get("export", {}) if export_config is not None else {}
    batch = int(settings.get("batch", 1))
    height = int(settings.get("height", 512))
    width = int(settings.get("width", 512))
    if height % 16 or width % 16:
        raise ValueError("static export dimensions must be multiples of 16")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.name}.tmp")
    generator = torch.Generator().manual_seed(20260726)
    sample = torch.rand(
        batch,
        model.input_channels,
        height,
        width,
        generator=generator,
    )
    # 相机嵌入允许负值；导出对照样例必须覆盖契约真实范围，而不是全通道 [0,1]。
    sample[:, 8:12] = sample[:, 8:12] * 2.0 - 1.0
    # 第 15 通道是有效区 mask；静态样例无 Pad，因此整张设为 1。
    sample[:, 15] = 1.0
    export_model = _StaticExportWrapper(model)
    torch.onnx.export(
        export_model,
        sample,
        temporary_output,
        input_names=[str(settings.get("input_name", "context_raw"))],
        output_names=[str(settings.get("output_name", "raw_residual"))],
        opset_version=int(settings.get("opset", 17)),
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("install the 'export' extra before exporting ONNX") from error
    onnx_model = onnx.load(str(temporary_output))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(temporary_output),
        providers=["CPUExecutionProvider"],
    )
    with torch.inference_mode():
        torch_output = model.forward_static(sample).cpu().numpy()
    runtime_output = session.run(
        [str(settings.get("output_name", "raw_residual"))],
        {str(settings.get("input_name", "context_raw")): sample.numpy()},
    )[0]
    # 同时记录绝对/相对误差；近零输出会放大相对误差，因此放行仍使用 atol+rtol。
    absolute_error = np.abs(torch_output - runtime_output)
    relative_error = absolute_error / np.maximum(np.abs(torch_output), 1e-6)
    max_absolute_error = float(absolute_error.max())
    max_relative_error = float(relative_error.max())
    np.testing.assert_allclose(
        runtime_output,
        torch_output,
        atol=float(settings.get("verify_atol", 1e-4)),
        rtol=float(settings.get("verify_rtol", 1e-3)),
    )
    # manifest 是模型文件的伴生证据，不把 ONNX 成功误报成麒麟 NPU 已验证。
    model_config_path = Path(model_config)
    checkpoint_path = Path(checkpoint)
    export_config_path = Path(export_config) if export_config is not None else None
    onnx_sha256 = sha256_file(temporary_output)
    onnx_audit = audit_onnx(temporary_output)
    # 只有 Checker、ORT 数值对照和结构审计全部成功后，才替换正式 ONNX。
    temporary_output.replace(output_path)
    manifest = {
        "format_version": 2,
        "context_contract_version": "raw16-v1",
        "model_name": "nafnet_raw_student",
        "created_unix": int(time()),
        "source": {
            "model_config": str(model_config_path),
            "model_config_sha256": sha256_file(model_config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "export_config": str(export_config_path) if export_config_path else None,
            "export_config_sha256": (
                sha256_file(export_config_path) if export_config_path else None
            ),
            "qat_config": str(qat_config_path) if qat_config_path else None,
            "qat_config_sha256": sha256_file(qat_config_path) if qat_config_path else None,
        },
        "input_shape": [batch, model.input_channels, height, width],
        "input_dtype": "float32-export; fp16-device-candidate",
        "input_channel_semantics": _INPUT_CHANNEL_SEMANTICS,
        "output_shape": [batch, model.output_channels, height, width],
        "output_dtype": "float32-export; fp16-device-candidate",
        "output_semantics": "four-channel canonical RAW residual",
        "physical_parameter_count": model.parameter_count(),
        "onnx_opset": int(settings.get("opset", 17)),
        "onnx_sha256": onnx_sha256,
        "toolchain": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "torch_onnx_exporter": "legacy_torchscript_dynamo_false",
        },
        "verification": {
            "onnx_checker": "passed",
            "onnxruntime_version": ort.__version__,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
            "onnx_audit": onnx_audit,
        },
        "quantization": (
            {
                "mode": "qat_qdq_int8",
                "converted_convolutions": qat_report.converted_convolutions,
                "excluded_convolutions": qat_report.excluded_convolutions,
                "simulated_int8_weight_ratio": qat_report.simulated_int8_weight_ratio,
                "quantize_linear_nodes": onnx_audit["operator_counts"].get(
                    "QuantizeLinear", 0
                ),
                "dequantize_linear_nodes": onnx_audit["operator_counts"].get(
                    "DequantizeLinear", 0
                ),
                "target_status": "requires exact HiAI CANN DDK QDQ conversion and profiler",
            }
            if qat_report is not None
            else {"mode": "float"}
        ),
        "deployment_status": "UNVERIFIED_UNTIL_TARGET_HIAI_CANN_PROFILING",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_text = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"
    temporary_manifest = manifest_path.with_name(f"{manifest_path.name}.tmp")
    temporary_manifest.write_text(manifest_text, encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return output_path
