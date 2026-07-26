from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from isp_ai_enhancement.config import load_yaml
from isp_ai_enhancement.models.factory import build_model_from_file


class _StaticExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, value: Tensor) -> Tensor:
        return self.model.forward_static(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
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
) -> Path:
    model = build_model_from_file(model_config)
    model.load_state_dict(load_checkpoint_state(checkpoint), strict=True)
    model.eval()
    settings = load_yaml(export_config).get("export", {}) if export_config is not None else {}
    batch = int(settings.get("batch", 1))
    height = int(settings.get("height", 512))
    width = int(settings.get("width", 512))
    if height % 16 or width % 16:
        raise ValueError("static export dimensions must be multiples of 16")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(20260726)
    sample = torch.rand(
        batch,
        model.input_channels,
        height,
        width,
        generator=generator,
    )
    sample[:, 15] = 1.0
    export_model = _StaticExportWrapper(model)
    torch.onnx.export(
        export_model,
        sample,
        output_path,
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
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        torch_output = model.forward_static(sample).cpu().numpy()
    runtime_output = session.run(
        [str(settings.get("output_name", "raw_residual"))],
        {str(settings.get("input_name", "context_raw")): sample.numpy()},
    )[0]
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
    manifest = {
        "format_version": 1,
        "model_name": "nafnet_raw_student",
        "input_shape": [batch, model.input_channels, height, width],
        "input_dtype": "float32-export; fp16-device-candidate",
        "output_shape": [batch, model.output_channels, height, width],
        "output_dtype": "float32-export; fp16-device-candidate",
        "output_semantics": "four-channel canonical RAW residual",
        "physical_parameter_count": model.parameter_count(),
        "onnx_opset": int(settings.get("opset", 17)),
        "onnx_sha256": sha256_file(output_path),
        "verification": {
            "onnx_checker": "passed",
            "onnxruntime_version": ort.__version__,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
        },
        "deployment_status": "UNVERIFIED_UNTIL_TARGET_HIAI_CANN_PROFILING",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
