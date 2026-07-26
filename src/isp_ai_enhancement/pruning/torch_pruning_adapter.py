"""Torch-Pruning 与 NAFNet SimpleGate 之间的结构化剪枝适配层。

Torch-Pruning 能根据 Autograd 图自动追踪卷积、逐通道卷积、乘法门控和 SCA
分支之间的通道依赖。但 NAFNet 的 SimpleGate 把通道等分成两半后逐元素相乘，
一个逻辑隐藏通道对应扩展卷积中的两个物理通道。因此本模块必须成对删除
``[i, i + hidden]``，并在剪枝后同步门控模块保存的静态通道数。
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

import torch
from torch import Tensor, nn

from isp_ai_enhancement.models.nafnet import ExpansionSpec, NAFBlock, NAFNetRaw

from .physical import PruningReport, _dw_importance, _ffn_importance, _top_indices


@dataclass(frozen=True)
class TorchPruningReport:
    """记录剪枝组数、组内联动操作数和门控通道删除数量。"""

    backend_version: str
    dependency_groups: int
    dependency_operations: int
    pruned_gate_units: int


def _installed_backend_version() -> str:
    """读取发行包元数据中的真实版本，规避模块 ``__version__`` 滞后问题。

    Torch-Pruning 1.6.1 的已安装发行包仍可能把 ``torch_pruning.__version__``
    报成 1.6.0，因此审计报告必须以 Python 包元数据为准。
    """

    try:
        return version("torch-pruning")
    except PackageNotFoundError:
        return "unknown"


def _require_torch_pruning() -> ModuleType:
    """延迟加载可选依赖，并给出可直接执行的中文安装提示。"""

    try:
        import torch_pruning
    except ImportError as error:
        raise RuntimeError(
            "Torch-Pruning 后端未安装；请执行 pip install -e '.[pruning]'"
        ) from error
    return torch_pruning


def _pruned_indices(score: Tensor, target_hidden: int) -> Tensor:
    """根据重要性保留高分通道，返回需要物理删除的原始通道索引。"""

    keep = _top_indices(score, target_hidden)
    remove_mask = torch.ones(score.numel(), dtype=torch.bool, device=score.device)
    remove_mask[keep] = False
    return torch.nonzero(remove_mask, as_tuple=False).flatten()


def _example_for_block(block: NAFBlock) -> Tensor:
    """构造依赖图所需的小尺寸输入，保持模型当前 device 和 dtype。"""

    reference = block.conv1.weight
    return torch.zeros(
        1,
        block.channels,
        8,
        8,
        device=reference.device,
        dtype=reference.dtype,
    )


def _prune_branch(
    block: NAFBlock,
    *,
    root: nn.Conv2d,
    current_hidden: int,
    target_hidden: int,
    score: Tensor,
) -> int:
    """让 DepGraph 删除一个 SimpleGate 分支中的成对扩展通道。

    返回依赖组中包含的联动操作数量，供审计报告确认剪枝不是单层 mask。
    """

    if target_hidden <= 0:
        raise ValueError("SimpleGate 剪枝后的隐藏通道数必须为正数")
    if target_hidden > current_hidden:
        raise ValueError(
            f"结构化剪枝不能扩张通道：{current_hidden} -> {target_hidden}"
        )
    if target_hidden == current_hidden:
        return 0

    tp = _require_torch_pruning()
    removed = _pruned_indices(score, target_hidden)
    # SimpleGate 左右两半必须删除同一逻辑位置，否则逐元素乘法语义会错位。
    paired = torch.cat((removed, removed + current_hidden)).tolist()
    with warnings.catch_warnings():
        # 主干 LayerNorm/beta/gamma 不参与本次扩展通道剪枝；过滤 DepGraph 的提示噪声。
        warnings.filterwarnings("ignore", message="Unwrapped parameters detected")
        graph = tp.DependencyGraph().build_dependency(
            block,
            example_inputs=_example_for_block(block),
        )
    group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=paired)
    if not graph.check_pruning_group(group):
        raise RuntimeError("Torch-Pruning 拒绝该依赖组，可能会把某层通道剪成零")
    operation_count = len(group)
    group.prune()
    return operation_count


def _prune_block(block: NAFBlock, target_hidden: int) -> tuple[int, int, int]:
    """按同一目标宽度剪掉 NAFBlock 的 depthwise/SCA 与 FFN 两个门控分支。"""

    groups = 0
    operations = 0
    removed_units = 0

    current_dw = block.dw_hidden_channels
    operations += _prune_branch(
        block,
        root=block.conv1,
        current_hidden=current_dw,
        target_hidden=target_hidden,
        score=_dw_importance(block),
    )
    if target_hidden != current_dw:
        groups += 1
        removed_units += current_dw - target_hidden
        # torch.split 使用此静态长度；DepGraph 不会自动修改普通 Python 属性。
        block.dw_hidden_channels = target_hidden
        block.gate1.hidden_channels = target_hidden

    current_ffn = block.ffn_hidden_channels
    operations += _prune_branch(
        block,
        root=block.conv4,
        current_hidden=current_ffn,
        target_hidden=target_hidden,
        score=_ffn_importance(block),
    )
    if target_hidden != current_ffn:
        groups += 1
        removed_units += current_ffn - target_hidden
        block.ffn_hidden_channels = target_hidden
        block.gate2.hidden_channels = target_hidden

    if block.conv1.out_channels != 2 * target_hidden:
        raise AssertionError("conv1 输出通道与 SimpleGate 目标宽度不一致")
    if block.sca_conv.in_channels != target_hidden:
        raise AssertionError("SCA 输入通道未被 DepGraph 同步裁剪")
    if block.conv5.in_channels != target_hidden:
        raise AssertionError("FFN 输出投影输入通道未被 DepGraph 同步裁剪")
    return groups, operations, removed_units


def _prune_stage(stage: nn.Sequential, widths: tuple[int, ...]) -> tuple[int, int, int]:
    """逐块执行剪枝，并强制配置宽度数量与实际 NAFBlock 数量一致。"""

    if len(stage) != len(widths):
        raise ValueError("目标扩展宽度数量与 NAFBlock 数量不一致")
    groups = 0
    operations = 0
    removed_units = 0
    for module, width in zip(stage, widths, strict=True):
        if not isinstance(module, NAFBlock):
            raise TypeError("Torch-Pruning 适配层只接受 NAFBlock stage")
        block_groups, block_operations, block_removed = _prune_block(module, width)
        groups += block_groups
        operations += block_operations
        removed_units += block_removed
    return groups, operations, removed_units


def torch_pruning_physical_prune(
    source: NAFNetRaw,
    expansion_spec: ExpansionSpec,
) -> tuple[NAFNetRaw, PruningReport, TorchPruningReport]:
    """使用 Torch-Pruning DepGraph 原地删通道，并返回独立的缩小模型副本。

    源模型不会被修改。依赖图构建阶段必须开启 Autograd，因此本函数不能被
    ``torch.no_grad`` 包裹；真正的参数选择只读取已训练权重，不产生训练梯度。
    """

    # 即使目标规格完全不变，也要验证用户显式选择的后端确实已经安装。
    _require_torch_pruning()
    target = copy.deepcopy(source)
    original_training = target.training
    target.eval()
    groups = 0
    operations = 0
    removed_units = 0

    for index, stage in enumerate(target.encoders, start=1):
        stage_groups, stage_operations, stage_removed = _prune_stage(
            stage, expansion_spec.stage(f"enc{index}")
        )
        groups += stage_groups
        operations += stage_operations
        removed_units += stage_removed
    stage_groups, stage_operations, stage_removed = _prune_stage(
        target.middle, expansion_spec.middle
    )
    groups += stage_groups
    operations += stage_operations
    removed_units += stage_removed
    for index, stage in enumerate(target.decoders, start=1):
        stage_groups, stage_operations, stage_removed = _prune_stage(
            stage, expansion_spec.stage(f"dec{index}")
        )
        groups += stage_groups
        operations += stage_operations
        removed_units += stage_removed

    target.expansion_spec = expansion_spec
    target.train(original_training)
    structural = PruningReport(source.parameter_count(), target.parameter_count())
    backend = TorchPruningReport(
        backend_version=_installed_backend_version(),
        dependency_groups=groups,
        dependency_operations=operations,
        pruned_gate_units=removed_units,
    )
    return target, structural, backend
