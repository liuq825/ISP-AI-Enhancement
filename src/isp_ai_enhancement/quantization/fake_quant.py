"""与具体 NPU 后端解耦的对称伪量化模块和卷积替换工具。

这里的 QAT 只模拟激活 per-tensor、权重 per-output-channel 的有符号整数量化误差，
不能代替目标 HiAI CANN DDK 生成的最终量化参数和算子落点报告。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SymmetricFakeQuant(nn.Module):
    """使用移动最大值观察器和直通估计器实现对称伪量化。"""

    def __init__(
        self,
        *,
        bits: int = 8,
        per_channel: bool = False,
        channel_axis: int = 0,
        momentum: float = 0.95,
    ) -> None:
        """配置位宽、按通道轴和观察器动量，并初始化运行时最大绝对值。"""

        super().__init__()
        if bits < 2 or bits > 16:
            raise ValueError("bits must be in [2, 16]")
        self.bits = bits
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.momentum = momentum
        self.register_buffer("max_abs", torch.ones(1))
        self.register_buffer(
            "observer_initialized",
            torch.tensor(False, dtype=torch.bool),
        )
        self.observer_enabled = True
        self.fake_quant_enabled = True

    def _observed_max(self, value: Tensor) -> Tensor:
        """计算当前批次的最大绝对值，按通道模式会保留指定轴。"""

        if not self.per_channel:
            return value.detach().abs().amax().reshape(1)
        dimensions = tuple(index for index in range(value.ndim) if index != self.channel_axis)
        return value.detach().abs().amax(dim=dimensions)

    def _reshape_scale(self, scale: Tensor, value: Tensor) -> Tensor:
        """把一维通道 scale 变形成可广播到权重张量的形状。"""

        if not self.per_channel:
            return scale
        shape = [1] * value.ndim
        shape[self.channel_axis] = scale.numel()
        return scale.reshape(shape)

    def forward(self, value: Tensor) -> Tensor:
        """更新观察器并执行量化-反量化，反向传播使用恒等直通梯度。"""

        observed = self._observed_max(value).clamp_min(1e-8)
        if self.observer_enabled and self.training:
            if self.max_abs.numel() != observed.numel():
                self.max_abs.resize_(observed.shape)
                self.observer_initialized.fill_(False)
            if not bool(self.observer_initialized.item()):
                # 首批必须直接采用真实范围；从固定 1.0 做 EMA 会严重浪费低幅 RAW 精度。
                self.max_abs.copy_(observed)
                self.observer_initialized.fill_(True)
            else:
                self.max_abs.mul_(self.momentum).add_(
                    observed, alpha=1.0 - self.momentum
                )
        if not self.fake_quant_enabled:
            return value
        if not bool(self.observer_initialized.item()):
            raise RuntimeError("fake quantization requires an initialized observer")
        quant_max = float((1 << (self.bits - 1)) - 1)
        scale = self._reshape_scale(self.max_abs.clamp_min(1e-8) / quant_max, value)
        quantized = torch.round(value / scale).clamp(-quant_max, quant_max) * scale
        return value + (quantized - value).detach()


class QATConv2d(nn.Conv2d):
    """在标准 Conv2d 前插入激活和权重伪量化，保持卷积参数接口不变。"""

    def __init__(
        self,
        *args,
        activation_bits: int = 8,
        weight_bits: int = 8,
        observer_momentum: float = 0.95,
        **kwargs,
    ):
        """构建卷积本体以及激活 per-tensor、权重 per-channel 两个观察器。"""

        super().__init__(*args, **kwargs)
        self.activation_fake_quant = SymmetricFakeQuant(
            bits=activation_bits,
            momentum=observer_momentum,
        )
        self.weight_fake_quant = SymmetricFakeQuant(
            bits=weight_bits,
            per_channel=True,
            channel_axis=0,
            momentum=observer_momentum,
        )

    @classmethod
    def from_float(
        cls,
        module: nn.Conv2d,
        *,
        activation_bits: int = 8,
        weight_bits: int = 8,
        observer_momentum: float = 0.95,
    ) -> QATConv2d:
        """从已有浮点卷积复制结构和权重，供模型原位 QAT 改写。"""

        result = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
            activation_bits=activation_bits,
            weight_bits=weight_bits,
            observer_momentum=observer_momentum,
        )
        result.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            result.bias.data.copy_(module.bias.data)
        result.train(module.training)
        return result

    def forward(self, value: Tensor) -> Tensor:
        """对输入和权重做伪量化后调用标准二维卷积。"""

        activation = self.activation_fake_quant(value)
        weight = self.weight_fake_quant(self.weight)
        return F.conv2d(
            activation,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


@dataclass(frozen=True)
class QATReport:
    """记录 QAT 转换/排除的卷积数量与权重元素覆盖率。"""

    converted_convolutions: int
    excluded_convolutions: int
    converted_weight_elements: int
    excluded_weight_elements: int

    @property
    def simulated_int8_ratio(self) -> float:
        """返回按卷积层数量估算的覆盖率，仅用于结构检查。"""

        total = self.converted_convolutions + self.excluded_convolutions
        return self.converted_convolutions / total if total else 0.0

    @property
    def simulated_int8_weight_ratio(self) -> float:
        """返回按卷积权重元素计算的伪 INT8 覆盖率。"""

        total = self.converted_weight_elements + self.excluded_weight_elements
        return self.converted_weight_elements / total if total else 0.0


def prepare_qat(
    model: nn.Module,
    *,
    activation_bits: int = 8,
    weight_bits: int = 8,
    observer_momentum: float = 0.95,
    exclude_modules: Iterable[str] = ("intro", "ending"),
) -> QATReport:
    """递归原位替换可量化卷积，并返回结构覆盖报告。"""
    excluded_prefixes = tuple(exclude_modules)
    converted = 0
    excluded = 0
    converted_weights = 0
    excluded_weights = 0

    def visit(parent: nn.Module, prefix: str = "") -> None:
        """深度优先遍历模块树，按完整名称匹配排除前缀。"""

        nonlocal converted, converted_weights, excluded, excluded_weights
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                if any(
                    full_name == item or full_name.startswith(f"{item}.")
                    for item in excluded_prefixes
                ):
                    excluded += 1
                    excluded_weights += child.weight.numel()
                else:
                    setattr(
                        parent,
                        name,
                        QATConv2d.from_float(
                            child,
                            activation_bits=activation_bits,
                            weight_bits=weight_bits,
                            observer_momentum=observer_momentum,
                        ),
                    )
                    converted += 1
                    converted_weights += child.weight.numel()
            else:
                visit(child, full_name)

    visit(model)
    return QATReport(
        converted,
        excluded,
        converted_weights,
        excluded_weights,
    )


def set_observer_enabled(model: nn.Module, enabled: bool) -> None:
    """统一启停全部观察器；冻结 scale 后继续训练时应传入 ``False``。"""

    for module in model.modules():
        if isinstance(module, SymmetricFakeQuant):
            module.observer_enabled = enabled


def set_fake_quant_enabled(model: nn.Module, enabled: bool) -> None:
    """统一启停伪量化误差注入，便于比较同一 QAT 权重的浮点/量化输出。"""

    for module in model.modules():
        if isinstance(module, SymmetricFakeQuant):
            module.fake_quant_enabled = enabled
