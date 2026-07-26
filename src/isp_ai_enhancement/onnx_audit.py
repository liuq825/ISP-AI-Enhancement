from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def audit_onnx(path: str | Path) -> dict[str, Any]:
    """Return a structural audit without claiming target NPU support."""
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("install the 'export' extra to audit ONNX") from error
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    operator_counts = Counter(node.op_type for node in model.graph.node)
    high_risk = {
        name: operator_counts[name]
        for name in (
            "ReduceMean",
            "Sqrt",
            "Reciprocal",
            "DepthToSpace",
            "Split",
            "Mul",
        )
        if operator_counts[name]
    }

    def value_shape(value_info) -> list[int | str]:
        dimensions = value_info.type.tensor_type.shape.dim
        return [
            dimension.dim_value
            if dimension.HasField("dim_value")
            else dimension.dim_param or "dynamic"
            for dimension in dimensions
        ]

    inputs = {item.name: value_shape(item) for item in model.graph.input}
    outputs = {item.name: value_shape(item) for item in model.graph.output}
    static_io = all(
        isinstance(dimension, int)
        for shape in (*inputs.values(), *outputs.values())
        for dimension in shape
    )
    dynamic_shape_operators = {
        name: operator_counts[name]
        for name in ("Shape", "Gather", "Mod", "Pad", "Slice")
        if operator_counts[name]
    }
    return {
        "ir_version": model.ir_version,
        "opset_imports": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": inputs,
        "outputs": outputs,
        "static_io": static_io,
        "operator_counts": dict(sorted(operator_counts.items())),
        "high_risk_operator_counts": high_risk,
        "dynamic_shape_operator_counts": dynamic_shape_operators,
        "target_support_status": "requires exact HiAI CANN DDK conversion log",
    }
