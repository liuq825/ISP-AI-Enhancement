from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm expressed with primitive ops for ONNX portability."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class SimpleGate(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

    def forward(self, value: Tensor) -> Tensor:
        left, right = torch.split(value, self.hidden_channels, dim=1)
        return left * right


class NAFBlock(nn.Module):
    """NAFBlock with independently rebuildable, paired expansion channels."""

    def __init__(
        self,
        channels: int,
        dw_hidden_channels: int | None = None,
        ffn_hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dw_hidden = dw_hidden_channels or channels
        ffn_hidden = ffn_hidden_channels or dw_hidden
        if dw_hidden <= 0 or ffn_hidden <= 0:
            raise ValueError("hidden channel counts must be positive")

        self.channels = channels
        self.dw_hidden_channels = dw_hidden
        self.ffn_hidden_channels = ffn_hidden
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, 2 * dw_hidden, 1)
        self.conv2 = nn.Conv2d(
            2 * dw_hidden,
            2 * dw_hidden,
            3,
            padding=1,
            groups=2 * dw_hidden,
        )
        self.gate1 = SimpleGate(dw_hidden)
        self.sca_pool = nn.AdaptiveAvgPool2d(1)
        self.sca_conv = nn.Conv2d(dw_hidden, dw_hidden, 1)
        self.conv3 = nn.Conv2d(dw_hidden, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, 2 * ffn_hidden, 1)
        self.gate2 = SimpleGate(ffn_hidden)
        self.conv5 = nn.Conv2d(ffn_hidden, channels, 1)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: Tensor) -> Tensor:
        branch = self.conv2(self.conv1(self.norm1(value)))
        branch = self.gate1(branch)
        branch = branch * self.sca_conv(self.sca_pool(branch))
        branch = self.dropout1(self.conv3(branch))
        residual = value + branch * self.beta

        branch = self.gate2(self.conv4(self.norm2(residual)))
        branch = self.dropout2(self.conv5(branch))
        return residual + branch * self.gamma


@dataclass(frozen=True)
class ExpansionSpec:
    """Post-gate channel counts for every NAFBlock stage."""

    enc1: tuple[int, ...]
    enc2: tuple[int, ...]
    enc3: tuple[int, ...]
    enc4: tuple[int, ...]
    middle: tuple[int, ...]
    dec1: tuple[int, ...]
    dec2: tuple[int, ...]
    dec3: tuple[int, ...]
    dec4: tuple[int, ...]

    @classmethod
    def baseline(
        cls,
        width: int,
        encoder_blocks: Sequence[int],
        middle_blocks: int,
        decoder_blocks: Sequence[int],
    ) -> ExpansionSpec:
        if len(encoder_blocks) != 4 or len(decoder_blocks) != 4:
            raise ValueError("the Kirin reference topology requires four encoder/decoder stages")
        encoder = [tuple([width * (2**i)] * count) for i, count in enumerate(encoder_blocks)]
        middle = tuple([width * 16] * middle_blocks)
        decoder_channels = [width * 8, width * 4, width * 2, width]
        decoder = [
            tuple([channels] * count)
            for channels, count in zip(decoder_channels, decoder_blocks, strict=True)
        ]
        return cls(*encoder, middle, *decoder)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Sequence[int]]) -> ExpansionSpec:
        names = (
            "enc1",
            "enc2",
            "enc3",
            "enc4",
            "middle",
            "dec1",
            "dec2",
            "dec3",
            "dec4",
        )
        missing = [name for name in names if name not in value]
        if missing:
            raise ValueError(f"expansion spec is missing stages: {missing}")
        return cls(*(tuple(int(item) for item in value[name]) for name in names))

    def stage(self, name: str) -> tuple[int, ...]:
        return getattr(self, name)

    def as_dict(self) -> dict[str, list[int]]:
        return {
            name: list(self.stage(name))
            for name in (
                "enc1",
                "enc2",
                "enc3",
                "enc4",
                "middle",
                "dec1",
                "dec2",
                "dec3",
                "dec4",
            )
        }


def reference_pruned_spec() -> ExpansionSpec:
    """16-aligned reference close to 15% physical parameter reduction.

    The original proposal's uniform middle width of 400 over-prunes this
    topology. Per-block middle widths keep the reference near the intended
    global target while leaving room for sensitivity-driven reallocation.
    """

    return ExpansionSpec(
        enc1=(32, 32),
        enc2=(64, 64),
        enc3=(112, 112, 112, 112),
        enc4=(224, 224, 224, 224, 224, 224, 224, 224),
        middle=(416, 416, 432, 432),
        dec1=(224, 224),
        dec2=(112, 112),
        dec3=(64, 64),
        dec4=(32, 32),
    )


class NAFNetRaw(nn.Module):
    """Four-level NAFNet that emits a four-channel RAW residual."""

    def __init__(
        self,
        input_channels: int = 16,
        output_channels: int = 4,
        width: int = 32,
        encoder_blocks: Sequence[int] = (2, 2, 4, 8),
        middle_blocks: int = 4,
        decoder_blocks: Sequence[int] = (2, 2, 2, 2),
        expansion_spec: ExpansionSpec | None = None,
    ) -> None:
        super().__init__()
        if len(encoder_blocks) != 4 or len(decoder_blocks) != 4:
            raise ValueError("NAFNetRaw requires exactly four encoder and decoder stages")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.width = width
        self.encoder_blocks = tuple(encoder_blocks)
        self.middle_blocks_count = middle_blocks
        self.decoder_blocks = tuple(decoder_blocks)
        self.expansion_spec = expansion_spec or ExpansionSpec.baseline(
            width, encoder_blocks, middle_blocks, decoder_blocks
        )
        self._validate_expansion_spec()

        self.intro = nn.Conv2d(input_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, output_channels, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for index, _count in enumerate(self.encoder_blocks, start=1):
            self.encoders.append(
                self._make_stage(channels, self.expansion_spec.stage(f"enc{index}"))
            )
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2

        self.middle = self._make_stage(channels, self.expansion_spec.middle)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index, _count in enumerate(self.decoder_blocks, start=1):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, 2 * channels, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                self._make_stage(channels, self.expansion_spec.stage(f"dec{index}"))
            )
        self.padder_size = 16

    @staticmethod
    def _make_stage(channels: int, hidden: Sequence[int]) -> nn.Sequential:
        return nn.Sequential(
            *[
                NAFBlock(
                    channels,
                    dw_hidden_channels=hidden_channels,
                    ffn_hidden_channels=hidden_channels,
                )
                for hidden_channels in hidden
            ]
        )

    def _validate_expansion_spec(self) -> None:
        expected = (
            *self.encoder_blocks,
            self.middle_blocks_count,
            *self.decoder_blocks,
        )
        names = (
            "enc1",
            "enc2",
            "enc3",
            "enc4",
            "middle",
            "dec1",
            "dec2",
            "dec3",
            "dec4",
        )
        for name, count in zip(names, expected, strict=True):
            values = self.expansion_spec.stage(name)
            if len(values) != count:
                raise ValueError(f"{name} has {len(values)} widths, expected {count}")
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} contains a non-positive width")

    def _pad(self, value: Tensor) -> Tensor:
        height, width = value.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(value, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    def _forward_core(self, value: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        features: dict[str, Tensor] = {}
        current = self.intro(value)
        skips: list[Tensor] = []
        for index, (encoder, down) in enumerate(
            zip(self.encoders, self.downs, strict=True), start=1
        ):
            current = encoder(current)
            features[f"enc{index}"] = current
            skips.append(current)
            current = down(current)
        current = self.middle(current)
        features["middle"] = current
        for index, (decoder, up, skip) in enumerate(
            zip(self.decoders, self.ups, reversed(skips), strict=True), start=1
        ):
            current = decoder(up(current) + skip)
            features[f"dec{index}"] = current
        residual = self.ending(current)
        return residual, features

    def forward_features(self, value: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if value.ndim != 4 or value.shape[1] != self.input_channels:
            raise ValueError(f"expected N×{self.input_channels}×H×W, received {tuple(value.shape)}")
        original_height, original_width = value.shape[-2:]
        residual, features = self._forward_core(self._pad(value))
        residual = residual[..., :original_height, :original_width]
        return residual, features

    def forward_static(self, value: Tensor) -> Tensor:
        """Forward for a prevalidated, 16-aligned static deployment shape."""
        residual, _ = self._forward_core(value)
        return residual

    def forward(self, value: Tensor) -> Tensor:
        residual, _ = self.forward_features(value)
        return residual

    def enhance(self, value: Tensor) -> Tensor:
        residual = self(value)
        return torch.clamp(value[:, : self.output_channels] + residual, 0.0, 1.0)

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (item for item in parameters if item.requires_grad)
        return sum(item.numel() for item in parameters)
