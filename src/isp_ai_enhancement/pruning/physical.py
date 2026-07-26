"""NAFNet 的确定性手工结构化剪枝基线。

本实现根据已训练权重的重要性选择 SimpleGate 的逻辑通道，重新构造较小网络
并复制相应权重。它不依赖第三方剪枝库，主要作为 Torch-Pruning 后端的
交叉验证基线以及受限环境下的回退方案。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from isp_ai_enhancement.models.nafnet import ExpansionSpec, NAFBlock, NAFNetRaw


@dataclass(frozen=True)
class PruningReport:
    """记录剪枝前后的真实参数量，并据此计算物理剪枝率。"""

    source_parameters: int
    target_parameters: int

    @property
    def pruning_ratio(self) -> float:
        """返回被实际移除的参数占源模型参数量的比例。"""

        return 1.0 - self.target_parameters / self.source_parameters


def _copy_parameter(target: Tensor, source: Tensor) -> None:
    """在复制前严格检查形状，防止广播造成静默权重损坏。"""

    if target.shape != source.shape:
        raise ValueError(f"shape mismatch: {tuple(target.shape)} != {tuple(source.shape)}")
    target.copy_(source)


def _paired_indices(indices: Tensor, source_hidden: int) -> Tensor:
    """把逻辑门控索引展开为 SimpleGate 左右两半的成对物理索引。"""

    return torch.cat((indices, indices + source_hidden))


def _top_indices(score: Tensor, count: int) -> Tensor:
    """选择重要性最高的指定数量通道，并恢复为递增索引便于复制。"""

    if count > score.numel():
        raise ValueError(f"cannot expand a block from {score.numel()} to {count} channels")
    selected = torch.topk(score, k=count, largest=True, sorted=False).indices
    return torch.sort(selected).values


def _dw_importance(block: NAFBlock) -> Tensor:
    """融合扩展卷积、SCA 与输出投影权重，估计第一门控分支重要性。"""

    hidden = block.dw_hidden_channels
    conv1 = block.conv1.weight.detach().square().mean(dim=(1, 2, 3))
    paired = conv1[:hidden] + conv1[hidden:]
    conv3 = block.conv3.weight.detach().square().mean(dim=(0, 2, 3))
    sca_in = block.sca_conv.weight.detach().square().mean(dim=(0, 2, 3))
    sca_out = block.sca_conv.weight.detach().square().mean(dim=(1, 2, 3))
    return paired + conv3 + 0.5 * (sca_in + sca_out)


def _ffn_importance(block: NAFBlock) -> Tensor:
    """融合 FFN 扩展和收缩卷积权重，估计第二门控分支重要性。"""

    hidden = block.ffn_hidden_channels
    conv4 = block.conv4.weight.detach().square().mean(dim=(1, 2, 3))
    paired = conv4[:hidden] + conv4[hidden:]
    conv5 = block.conv5.weight.detach().square().mean(dim=(0, 2, 3))
    return paired + conv5


@torch.no_grad()
def _copy_block(source: NAFBlock, target: NAFBlock) -> None:
    """按重要性把一个源 NAFBlock 的可保留权重复制到较窄目标块。"""

    for name in ("weight", "bias"):
        _copy_parameter(getattr(target.norm1, name), getattr(source.norm1, name))
        _copy_parameter(getattr(target.norm2, name), getattr(source.norm2, name))
    _copy_parameter(target.beta, source.beta)
    _copy_parameter(target.gamma, source.gamma)

    # SimpleGate 两半必须选取相同逻辑索引，否则逐元素乘法会错配语义。
    dw_indices = _top_indices(_dw_importance(source), target.dw_hidden_channels)
    dw_paired = _paired_indices(dw_indices, source.dw_hidden_channels)
    target.conv1.weight.copy_(source.conv1.weight[dw_paired])
    target.conv1.bias.copy_(source.conv1.bias[dw_paired])
    target.conv2.weight.copy_(source.conv2.weight[dw_paired])
    target.conv2.bias.copy_(source.conv2.bias[dw_paired])
    target.sca_conv.weight.copy_(source.sca_conv.weight[dw_indices][:, dw_indices])
    target.sca_conv.bias.copy_(source.sca_conv.bias[dw_indices])
    target.conv3.weight.copy_(source.conv3.weight[:, dw_indices])
    target.conv3.bias.copy_(source.conv3.bias)

    ffn_indices = _top_indices(_ffn_importance(source), target.ffn_hidden_channels)
    ffn_paired = _paired_indices(ffn_indices, source.ffn_hidden_channels)
    target.conv4.weight.copy_(source.conv4.weight[ffn_paired])
    target.conv4.bias.copy_(source.conv4.bias[ffn_paired])
    target.conv5.weight.copy_(source.conv5.weight[:, ffn_indices])
    target.conv5.bias.copy_(source.conv5.bias)


@torch.no_grad()
def _copy_stage(source: nn.Sequential, target: nn.Sequential) -> None:
    """逐块迁移一个 stage，并要求源/目标块数完全一致。"""

    if len(source) != len(target):
        raise ValueError("source and target stages have different block counts")
    for source_block, target_block in zip(source, target, strict=True):
        if not isinstance(source_block, NAFBlock) or not isinstance(target_block, NAFBlock):
            raise TypeError("physical pruning expects NAFBlock stages")
        _copy_block(source_block, target_block)


@torch.no_grad()
def physical_prune(
    source: NAFNetRaw,
    expansion_spec: ExpansionSpec,
) -> tuple[NAFNetRaw, PruningReport]:
    """重建更小计算图并迁移成对门控通道，源模型保持不变。

    目标模型的主干宽度和块数不变，只按 ``expansion_spec`` 缩减各 NAFBlock
    内部扩展通道，因此残差连接与编码器/解码器接口无需额外对齐。
    """

    target = NAFNetRaw(
        input_channels=source.input_channels,
        output_channels=source.output_channels,
        width=source.width,
        encoder_blocks=source.encoder_blocks,
        middle_blocks=source.middle_blocks_count,
        decoder_blocks=source.decoder_blocks,
        expansion_spec=expansion_spec,
    ).to(device=source.intro.weight.device, dtype=source.intro.weight.dtype)
    target.intro.load_state_dict(source.intro.state_dict())
    target.ending.load_state_dict(source.ending.state_dict())
    for source_module, target_module in zip(source.downs, target.downs, strict=True):
        target_module.load_state_dict(source_module.state_dict())
    for source_module, target_module in zip(source.ups, target.ups, strict=True):
        target_module.load_state_dict(source_module.state_dict())
    for source_stage, target_stage in zip(source.encoders, target.encoders, strict=True):
        _copy_stage(source_stage, target_stage)
    _copy_stage(source.middle, target.middle)
    for source_stage, target_stage in zip(source.decoders, target.decoders, strict=True):
        _copy_stage(source_stage, target_stage)
    report = PruningReport(source.parameter_count(), target.parameter_count())
    return target, report
