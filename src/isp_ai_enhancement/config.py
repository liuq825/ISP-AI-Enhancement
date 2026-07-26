from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject ambiguous or empty top-level documents."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a YAML mapping at the top level")
    return value
