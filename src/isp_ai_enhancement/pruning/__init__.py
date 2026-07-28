"""NAFNet 物理结构化剪枝的公共接口。"""

from .physical import PruningReport, physical_prune, stage_hidden_retention
from .torch_pruning_adapter import TorchPruningReport, torch_pruning_physical_prune
from .workflow import prune_checkpoint

__all__ = [
    "PruningReport",
    "TorchPruningReport",
    "physical_prune",
    "prune_checkpoint",
    "stage_hidden_retention",
    "torch_pruning_physical_prune",
]
