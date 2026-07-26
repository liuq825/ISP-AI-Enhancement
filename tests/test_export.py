"""验证静态 ONNX 导出、ORT 数值对照和模型清单证据。"""

import json
from pathlib import Path

import torch

from isp_ai_enhancement.export import export_onnx
from isp_ai_enhancement.models.nafnet import NAFNetRaw
from isp_ai_enhancement.onnx_audit import audit_onnx
from isp_ai_enhancement.quantization.fake_quant import prepare_qat


def test_export_writes_static_verified_manifest(tmp_path: Path) -> None:
    """极小模型导出后应为静态图，并记录完整输入语义与来源哈希。"""

    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "model:\n"
        "  name: nafnet_raw\n"
        "  input_channels: 16\n"
        "  output_channels: 4\n"
        "  width: 4\n"
        "  encoder_blocks: [1, 1, 1, 1]\n"
        "  middle_blocks: 1\n"
        "  decoder_blocks: [1, 1, 1, 1]\n"
        "  expansion_spec: baseline\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": NAFNetRaw(
                width=4,
                encoder_blocks=(1, 1, 1, 1),
                middle_blocks=1,
                decoder_blocks=(1, 1, 1, 1),
            ).state_dict()
        },
        checkpoint,
    )
    export_config = tmp_path / "export.yaml"
    export_config.write_text(
        "export:\n"
        "  opset: 17\n"
        "  batch: 1\n"
        "  height: 16\n"
        "  width: 16\n"
        "  input_name: context_raw\n"
        "  output_name: raw_residual\n"
        "  verify_atol: 0.0001\n"
        "  verify_rtol: 0.001\n",
        encoding="utf-8",
    )
    output = export_onnx(
        model_config=model_config,
        checkpoint=checkpoint,
        output=tmp_path / "model.onnx",
        export_config=export_config,
    )
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    audit = audit_onnx(output)
    assert manifest["format_version"] == 2
    assert manifest["context_contract_version"] == "raw16-v1"
    assert len(manifest["input_channel_semantics"]) == 16
    assert len(manifest["onnx_sha256"]) == 64
    assert manifest["verification"]["onnx_checker"] == "passed"
    assert audit["static_io"]
    assert audit["dynamic_shape_operator_counts"] == {}


def test_qat_checkpoint_exports_standard_qdq_nodes(tmp_path: Path) -> None:
    """QAT checkpoint 应先重建观察器，再导出标准 Quantize/Dequantize 节点。"""

    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "model:\n"
        "  name: nafnet_raw\n"
        "  input_channels: 16\n"
        "  output_channels: 4\n"
        "  width: 4\n"
        "  encoder_blocks: [1, 1, 1, 1]\n"
        "  middle_blocks: 1\n"
        "  decoder_blocks: [1, 1, 1, 1]\n"
        "  expansion_spec: baseline\n",
        encoding="utf-8",
    )
    qat_config = tmp_path / "qat.yaml"
    qat_config.write_text(
        "qat:\n"
        "  activation_bits: 8\n"
        "  weight_bits: 8\n"
        "  observer_momentum: 0.95\n"
        "  exclude_modules: [intro, ending]\n",
        encoding="utf-8",
    )
    model = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    )
    prepare_qat(model)
    model.train()
    model(torch.randn(1, 16, 16, 16))
    checkpoint = tmp_path / "qat.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    export_config = tmp_path / "export.yaml"
    export_config.write_text(
        "export:\n"
        "  opset: 17\n"
        "  batch: 1\n"
        "  height: 16\n"
        "  width: 16\n"
        "  input_name: context_raw\n"
        "  output_name: raw_residual\n"
        "  verify_atol: 0.0001\n"
        "  verify_rtol: 0.001\n",
        encoding="utf-8",
    )

    output = export_onnx(
        model_config=model_config,
        checkpoint=checkpoint,
        output=tmp_path / "qat.onnx",
        export_config=export_config,
        qat_config=qat_config,
    )
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    audit = audit_onnx(output)
    assert manifest["quantization"]["mode"] == "qat_qdq_int8"
    assert manifest["quantization"]["converted_convolutions"] > 0
    assert audit["operator_counts"]["QuantizeLinear"] > 0
    assert audit["operator_counts"]["DequantizeLinear"] > 0
    assert audit["dynamic_shape_operator_counts"] == {}
