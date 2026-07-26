"""项目 YAML 配置的统一 UTF-8 读取与顶层类型校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML mapping，并拒绝空文档、列表等歧义顶层结构。"""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a YAML mapping at the top level")
    return value
