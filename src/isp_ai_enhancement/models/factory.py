from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isp_ai_enhancement.config import load_yaml

from .nafnet import ExpansionSpec, NAFNetRaw


def build_model(config: Mapping[str, Any]) -> NAFNetRaw:
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
    return build_model(load_yaml(path))
