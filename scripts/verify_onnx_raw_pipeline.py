"""用 ONNX Runtime 验证 16 通道 RAW 模型的端到端输入输出契约。

该脚本接受两类输入：
1. 单帧 Sensor Bayer 马赛克（``--raw-layout mosaic``），按 CFA 打包；
2. HDR/MFNR 已融合的四通道 canonical packed RAW（``--raw-layout packed``）。

输出是网络残差与输入 packed RAW 相加后的归一化 canonical RAW；可选择再还原为 Bayer
马赛克，交给后续 Demosaic 或传统 ISP。HDR/MFNR 的融合本身应由上游 ISP 完成，融合置信度
与运动鬼影图通过第 7、8 通道显式传入，不能在本脚本中伪造为真实融合结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from isp_ai_enhancement.data.context import (
    ContextBuilder,
    RawMetadata,
    canonical_pack_bayer,
    load_context_config,
)

_CFA_POSITIONS = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (1, 1), (0, 0), (0, 1)),
    "BGGR": ((1, 1), (1, 0), (0, 0), (0, 1)),
}


def _normalize_sensor_values(
    value: np.ndarray, black_level: float, white_level: float
) -> np.ndarray:
    """以 Sensor 黑白电平将数组规范到 [0, 1]，拒绝错误的电平范围。"""

    if white_level <= black_level:
        raise ValueError("white-level must be greater than black-level")
    normalized = (value.astype(np.float32) - black_level) / (white_level - black_level)
    if not np.isfinite(normalized).all():
        raise ValueError("RAW input contains NaN or infinity")
    return np.clip(normalized, 0.0, 1.0)


def _canonical_unpack_bayer(packed: np.ndarray, cfa_pattern: str) -> np.ndarray:
    """把 canonical ``[R, Gr, Gb, B]`` packed RAW 还原为指定 CFA 的 Bayer 马赛克。"""

    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError("packed output must have shape 4×H×W")
    pattern = cfa_pattern.upper()
    if pattern not in _CFA_POSITIONS:
        raise ValueError(f"unsupported CFA pattern: {cfa_pattern}")
    height, width = packed.shape[1:]
    mosaic = np.empty((height * 2, width * 2), dtype=np.float32)
    for channel, (row, column) in enumerate(_CFA_POSITIONS[pattern]):
        mosaic[row::2, column::2] = packed[channel]
    return mosaic


def _load_packed_raw(args: argparse.Namespace) -> torch.Tensor:
    """读取、归一化并转为单批次 ``1×4×H×W`` canonical packed RAW。"""

    if args.generate_smoke:
        # 固定合成 Bayer 只用于脚本自检；它不是相机采集数据，也不用于画质结论。
        height = width = 128
        rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        source = ((rows * 17 + columns * 13) % 1024).astype(np.float32)
        layout = "mosaic"
    else:
        source = np.load(args.raw_npy, allow_pickle=False)
        layout = args.raw_layout
    normalized = _normalize_sensor_values(source, args.black_level, args.white_level)
    if layout == "mosaic":
        if normalized.ndim != 2:
            raise ValueError("mosaic RAW must have shape H×W")
        packed = canonical_pack_bayer(torch.from_numpy(normalized), args.cfa_pattern)
        return packed.unsqueeze(0)
    if normalized.ndim != 3:
        raise ValueError("packed RAW must have shape 4×H×W or H×W×4")
    if normalized.shape[0] == 4:
        canonical = normalized
    elif normalized.shape[-1] == 4:
        canonical = np.moveaxis(normalized, -1, 0)
    else:
        raise ValueError("packed RAW must contain exactly four channels")
    return torch.from_numpy(np.ascontiguousarray(canonical)).unsqueeze(0)


def _load_condition_map(
    path: str | None, height: int, width: int, name: str
) -> torch.Tensor | None:
    """读取融合置信度或鬼影图，统一为 ``1×1×H×W`` 并严格校验分辨率。"""

    if path is None:
        return None
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3 and value.shape[0] == 1:
        value = value[:, None]
    elif value.ndim != 4:
        raise ValueError(f"{name} map must have shape H×W, 1×H×W, or 1×1×H×W")
    if value.shape != (1, 1, height, width):
        raise ValueError(
            f"{name} map shape {value.shape} does not match packed RAW {(height, width)}"
        )
    return torch.from_numpy(np.clip(value, 0.0, 1.0))


def _static_shape(session: ort.InferenceSession) -> tuple[int, int, int, int]:
    """读取 ONNX 静态输入形状，拒绝动态或非 raw16-v1 图以防 ABI 用错。"""

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("model must have exactly one input")
    shape = inputs[0].shape
    if len(shape) != 4 or any(not isinstance(item, int) for item in shape):
        raise ValueError(f"model input must be static N×C×H×W, got {shape}")
    batch, channels, height, width = (int(item) for item in shape)
    if batch != 1 or channels != 16:
        raise ValueError(f"model input must be 1×16×H×W, got {shape}")
    return batch, channels, height, width


def main() -> int:
    """解析参数、构建 16 通道上下文、执行 ONNX 并保存 ISP 可消费的 RAW。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="QAT 或 FP32 ONNX 模型路径")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-npy", help="Sensor mosaic 或已融合 packed RAW 的 .npy 文件")
    source.add_argument("--generate-smoke", action="store_true", help="生成确定性 Bayer 输入自检")
    parser.add_argument("--raw-layout", choices=("mosaic", "packed"), default="mosaic")
    parser.add_argument("--output", required=True, help="输出归一化 RAW .npy 文件")
    parser.add_argument("--output-layout", choices=("packed", "mosaic"), default="mosaic")
    parser.add_argument("--context-config", default="configs/context.yaml")
    parser.add_argument("--sensor-id", default="smoke_sensor")
    parser.add_argument("--mode", choices=("single", "hdr", "mfnr"), default="single")
    parser.add_argument("--cfa-pattern", default="RGGB")
    parser.add_argument("--black-level", type=float, default=0.0)
    parser.add_argument("--white-level", type=float, default=1.0)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument("--exposure-ratio", type=float, default=1.0)
    parser.add_argument("--wb-rg", type=float, default=1.0)
    parser.add_argument("--wb-bg", type=float, default=1.0)
    parser.add_argument("--fusion-confidence-npy")
    parser.add_argument("--motion-ghost-npy")
    parser.add_argument("--report", help="可选 JSON 验证报告路径")
    args = parser.parse_args()

    packed = _load_packed_raw(args)
    _, _, packed_height, packed_width = packed.shape
    confidence = _load_condition_map(
        args.fusion_confidence_npy,
        packed_height,
        packed_width,
        "fusion_confidence",
    )
    ghost = _load_condition_map(args.motion_ghost_npy, packed_height, packed_width, "motion_ghost")
    metadata = RawMetadata(
        sensor_id=args.sensor_id,
        mode=args.mode,
        noise_sigma=args.noise_sigma,
        exposure_ratio=args.exposure_ratio,
        wb_rg=args.wb_rg,
        wb_bg=args.wb_bg,
    )
    context = ContextBuilder(load_context_config(args.context_config)).build(
        packed, metadata, fusion_confidence=confidence, motion_ghost=ghost
    ).numpy()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    _, _, expected_height, expected_width = _static_shape(session)
    if (packed_height, packed_width) != (expected_height, expected_width):
        raise ValueError(
            f"packed RAW is {packed_height}×{packed_width}, but model requires "
            f"{expected_height}×{expected_width}; tile or pad upstream before inference"
        )
    output_name = session.get_outputs()[0].name
    residual = session.run([output_name], {session.get_inputs()[0].name: context})[0]
    if residual.shape != (1, 4, packed_height, packed_width):
        raise ValueError(f"unexpected model output shape: {residual.shape}")
    enhanced = np.clip(context[:, :4] + residual, 0.0, 1.0)[0].astype(np.float32)
    result = (
        enhanced
        if args.output_layout == "packed"
        else _canonical_unpack_bayer(enhanced, args.cfa_pattern)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)
    report = {
        "model": str(args.model),
        "mode": args.mode,
        "input_layout": args.raw_layout if not args.generate_smoke else "generated_mosaic",
        "context_shape": list(context.shape),
        "output_layout": args.output_layout,
        "output_shape": list(result.shape),
        "output_range": [float(result.min()), float(result.max())],
        "next_stage": "demosaic_or_traditional_isp",
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
