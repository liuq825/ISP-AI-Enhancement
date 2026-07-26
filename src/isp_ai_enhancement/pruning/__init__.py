"""NAFNet 物理结构化剪枝的公共接口。"""

from .physical import PruningReport, physical_prune
from .torch_pruning_adapter import TorchPruningReport, torch_pruning_physical_prune

__all__ = [
    "PruningReport",
    "TorchPruningReport",
    "physical_prune",
    "torch_pruning_physical_prune",
]
