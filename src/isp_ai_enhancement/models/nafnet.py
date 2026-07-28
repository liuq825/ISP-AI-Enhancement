"""面向 RAW 域降噪增强的 NAFNet 主干、可剪枝通道规格和静态部署前向。

本文件只使用卷积、逐元素运算、PixelShuffle 等端侧常见算子。网络输入遵循
``N×16×H×W`` 上下文契约，输出四通道 canonical RAW 残差；最终加回与裁剪由
``enhance`` 或宿主 ISP 完成。每个 NAFBlock 的门控隐藏宽度均可独立重建，
以便真正减少参数和 MAC，而不是仅给权重乘零掩码。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """使用基础算子实现二维逐像素通道归一化，避免依赖专用 LayerNorm 算子。"""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        """创建可学习的逐通道缩放和偏移参数，张量形状固定为 ``1×C×1×1``。"""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        """在通道维计算均值/方差，保持批次和空间尺寸不变。"""

        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class SimpleGate(nn.Module):
    """把 ``2C`` 通道等分后逐元素相乘，替代传统激活函数。"""

    def __init__(self, hidden_channels: int) -> None:
        """保存静态半通道长度；结构化剪枝后必须同步更新该值。"""

        super().__init__()
        self.hidden_channels = hidden_channels

    def forward(self, value: Tensor) -> Tensor:
        """按固定长度拆成左右两半并相乘，输出通道数由 ``2C`` 降为 ``C``。"""

        left, right = torch.split(value, self.hidden_channels, dim=1)
        return left * right


class NAFBlock(nn.Module):
    """包含 depthwise/SCA 与 FFN 两条残差分支的可物理剪枝 NAFBlock。"""

    def __init__(
        self,
        channels: int,
        dw_hidden_channels: int | None = None,
        ffn_hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """按逻辑门控宽度构建卷积；扩展卷积实际输出为隐藏宽度的两倍。"""

        super().__init__()
        dw_hidden = dw_hidden_channels or channels
        ffn_hidden = ffn_hidden_channels or dw_hidden
        if dw_hidden <= 0 or ffn_hidden <= 0:
            raise ValueError("hidden channel counts must be positive")

        self.channels = channels
        self.dw_hidden_channels = dw_hidden
        self.ffn_hidden_channels = ffn_hidden
        # 第一分支：1×1 扩展 → depthwise 3×3 → SimpleGate → SCA → 1×1 投影。
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

        # 第二分支是轻量 FFN：1×1 扩展 → SimpleGate → 1×1 投影。
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, 2 * ffn_hidden, 1)
        self.gate2 = SimpleGate(ffn_hidden)
        self.conv5 = nn.Conv2d(ffn_hidden, channels, 1)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: Tensor) -> Tensor:
        """执行两次带零初始化缩放的残差更新，初始状态接近恒等映射。"""

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
    """逐 NAFBlock 记录 SimpleGate 之后的逻辑隐藏通道数。"""

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
        """生成未剪枝基线宽度；每个 stage 默认等于该层主干通道数。"""

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
        """从 YAML 映射构造强类型规格，并拒绝缺失任一网络 stage。"""

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
        """按名称读取一个 stage 的逐块隐藏宽度。"""

        return getattr(self, name)

    def as_dict(self) -> dict[str, list[int]]:
        """转换为可写入 YAML/JSON 的普通列表映射。"""

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


def structure_aware_pruned_spec() -> ExpansionSpec:
    """返回 `[2,2,6,8]` Student 的结构感知约 15% 参考规格。

    分配规则不是给每个 block 统一乘 0.85：高分辨率 ``enc1/enc2/dec3/dec4`` 完整
    保留；``enc3/enc4`` 的 stage 首尾块比内部块更宽，以保护降采样前后和 skip
    接口；Middle 与深层 Decoder 承担更多压缩。所有隐藏宽度保持 16 对齐，当前物理
    参数剪枝率约 14.954%。真实 P1/P2/P3 仍须结合已训练权重和逐域敏感度复核。
    """

    return ExpansionSpec(
        enc1=(32, 32),
        enc2=(64, 64),
        enc3=(128, 112, 112, 112, 112, 128),
        enc4=(256, 224, 208, 208, 208, 208, 224, 256),
        middle=(448, 400, 400, 432),
        dec1=(224, 240),
        dec2=(112, 128),
        dec3=(64, 64),
        dec4=(32, 32),
    )


def reference_pruned_spec() -> ExpansionSpec:
    """兼容旧调用名，返回当前结构感知 15% 参考规格。"""

    return structure_aware_pruned_spec()


class NAFNetRaw(nn.Module):
    """四级编码器/解码器 NAFNet，输入 16 通道上下文并输出四通道 RAW 残差。"""

    def __init__(
        self,
        input_channels: int = 16,
        output_channels: int = 4,
        width: int = 32,
        encoder_blocks: Sequence[int] = (2, 2, 6, 8),
        middle_blocks: int = 4,
        decoder_blocks: Sequence[int] = (2, 2, 2, 2),
        expansion_spec: ExpansionSpec | None = None,
    ) -> None:
        """构建固定四级拓扑，并校验逐块扩展宽度与 block 数完全一致。"""

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

        # intro/ending 保留 FP16/FP32 候选，QAT 默认不量化以保护输入与输出精度。
        self.intro = nn.Conv2d(input_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, output_channels, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        # 每级编码后用 stride=2 卷积降采样，主干通道数翻倍。
        for index, _count in enumerate(self.encoder_blocks, start=1):
            self.encoders.append(
                self._make_stage(channels, self.expansion_spec.stage(f"enc{index}"))
            )
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2

        self.middle = self._make_stage(channels, self.expansion_spec.middle)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # 解码端先 1×1 扩展，再用 PixelShuffle 上采样并与同尺度 skip 相加。
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
        """按隐藏宽度列表创建一个 NAFBlock 序列，每个 block 可独立剪枝。"""

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
        """检查九个 stage 的宽度数量、正值约束和模型拓扑是否一致。"""

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
        """仅在右侧和底部补零，使任意输入尺寸对齐四次 2 倍下采样。"""

        height, width = value.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(value, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    def _forward_core(self, value: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """执行无动态 Pad 的核心图，并返回蒸馏所需的多尺度特征。"""

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
        """校验输入、自动补齐到 16 倍数，并把残差裁回原始 packed RAW 尺寸。"""

        if value.ndim != 4 or value.shape[1] != self.input_channels:
            raise ValueError(f"expected N×{self.input_channels}×H×W, received {tuple(value.shape)}")
        original_height, original_width = value.shape[-2:]
        residual, features = self._forward_core(self._pad(value))
        residual = residual[..., :original_height, :original_width]
        return residual, features

    def forward_static(self, value: Tensor) -> Tensor:
        """执行已由宿主验证为 16 对齐的静态前向，避免 ONNX 中出现动态 shape 子图。"""

        residual, _ = self._forward_core(value)
        return residual

    def forward(self, value: Tensor) -> Tensor:
        """返回任意空间尺寸输入对应的四通道 RAW 残差。"""

        residual, _ = self.forward_features(value)
        return residual

    def enhance(self, value: Tensor) -> Tensor:
        """把预测残差加到输入前四通道并裁剪到线性 RAW 合法范围 ``[0,1]``。"""

        residual = self(value)
        return torch.clamp(value[:, : self.output_channels] + residual, 0.0, 1.0)

    def parameter_count(self, trainable_only: bool = False) -> int:
        """统计物理张量元素数量；该值用于剪枝 Gate，不能用稀疏零值替代。"""

        parameters = self.parameters()
        if trainable_only:
            parameters = (item for item in parameters if item.requires_grad)
        return sum(item.numel() for item in parameters)
