"""从版本化 YAML 配置构建 NAFNetRaw，统一训练、剪枝和导出的模型入口。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isp_ai_enhancement.config import load_yaml

from .nafnet import ExpansionSpec, NAFNetRaw


def build_model(config: Mapping[str, Any]) -> NAFNetRaw:
    """解析模型映射并实例化 NAFNetRaw，支持 baseline 或显式逐块宽度。"""

    model_config = config.get("model", config)
    expansion_value = model_config.get("expansion_spec", "baseline")
    expansion = (
        None if expansion_value == "baseline" else ExpansionSpec.from_mapping(expansion_value)
    )
    return NAFNetRaw(
        input_channels=int(model_config.get("input_channels", 16)),
        output_channels=int(model_config.get("output_channels", 4)),
        width=int(model_config.get("width", 32)),
        encoder_blocks=tuple(model_config.get("encoder_blocks", (2, 2, 4, 8))),
        middle_blocks=int(model_config.get("middle_blocks", 4)),
        decoder_blocks=tuple(model_config.get("decoder_blocks", (2, 2, 2, 2))),
        expansion_spec=expansion,
    )


def build_model_from_file(path: str | Path) -> NAFNetRaw:
    """读取 UTF-8 YAML 文件后构建模型，避免各命令重复实现配置解析。"""

    return build_model(load_yaml(path))
