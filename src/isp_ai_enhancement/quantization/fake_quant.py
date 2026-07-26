from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SymmetricFakeQuant(nn.Module):
    """Small backend-neutral fake quantizer with a straight-through estimator."""

    def __init__(
        self,
        *,
        bits: int = 8,
        per_channel: bool = False,
        channel_axis: int = 0,
        momentum: float = 0.95,
    ) -> None:
        super().__init__()
        if bits < 2 or bits > 16:
            raise ValueError("bits must be in [2, 16]")
        self.bits = bits
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.momentum = momentum
        self.register_buffer("max_abs", torch.ones(1))
        self.observer_enabled = True
        self.fake_quant_enabled = True

    def _observed_max(self, value: Tensor) -> Tensor:
        if not self.per_channel:
            return value.detach().abs().amax().reshape(1)
        dimensions = tuple(index for index in range(value.ndim) if index != self.channel_axis)
        return value.detach().abs().amax(dim=dimensions)

    def _reshape_scale(self, scale: Tensor, value: Tensor) -> Tensor:
        if not self.per_channel:
            return scale
        shape = [1] * value.ndim
        shape[self.channel_axis] = scale.numel()
        return scale.reshape(shape)

    def forward(self, value: Tensor) -> Tensor:
        observed = self._observed_max(value)
        if self.max_abs.numel() != observed.numel():
            self.max_abs.resize_(observed.shape).copy_(observed.clamp_min(1e-8))
        elif self.observer_enabled:
            self.max_abs.mul_(self.momentum).add_(
                observed.clamp_min(1e-8), alpha=1.0 - self.momentum
            )
        if not self.fake_quant_enabled:
            return value
        quant_max = float((1 << (self.bits - 1)) - 1)
        scale = self._reshape_scale(self.max_abs.clamp_min(1e-8) / quant_max, value)
        quantized = torch.round(value / scale).clamp(-quant_max, quant_max) * scale
        return value + (quantized - value).detach()


class QATConv2d(nn.Conv2d):
    """Conv2d with activation and per-output-channel weight fake quantization."""

    def __init__(self, *args, quant_bits: int = 8, observer_momentum: float = 0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_fake_quant = SymmetricFakeQuant(bits=quant_bits, momentum=observer_momentum)
        self.weight_fake_quant = SymmetricFakeQuant(
            bits=quant_bits,
            per_channel=True,
            channel_axis=0,
            momentum=observer_momentum,
        )

    @classmethod
    def from_float(
        cls,
        module: nn.Conv2d,
        *,
        bits: int = 8,
        observer_momentum: float = 0.95,
    ) -> QATConv2d:
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
            quant_bits=bits,
            observer_momentum=observer_momentum,
        )
        result.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            result.bias.data.copy_(module.bias.data)
        return result

    def forward(self, value: Tensor) -> Tensor:
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
    converted_convolutions: int
    excluded_convolutions: int

    @property
    def simulated_int8_ratio(self) -> float:
        total = self.converted_convolutions + self.excluded_convolutions
        return self.converted_convolutions / total if total else 0.0


def prepare_qat(
    model: nn.Module,
    *,
    bits: int = 8,
    observer_momentum: float = 0.95,
    exclude_modules: Iterable[str] = ("intro", "ending"),
) -> QATReport:
    """Replace eligible convolutions in-place and return a structural report."""
    excluded_prefixes = tuple(exclude_modules)
    converted = 0
    excluded = 0

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted, excluded
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                if any(
                    full_name == item or full_name.startswith(f"{item}.")
                    for item in excluded_prefixes
                ):
                    excluded += 1
                else:
                    setattr(
                        parent,
                        name,
                        QATConv2d.from_float(
                            child,
                            bits=bits,
                            observer_momentum=observer_momentum,
                        ),
                    )
                    converted += 1
            else:
                visit(child, full_name)

    visit(model)
    return QATReport(converted, excluded)


def set_observer_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, SymmetricFakeQuant):
            module.observer_enabled = enabled


def set_fake_quant_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, SymmetricFakeQuant):
            module.fake_quant_enabled = enabled
