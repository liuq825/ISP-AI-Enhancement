"""验证带 halo 的分块推理拼接与整帧结果一致。"""

import torch
from torch import nn

from isp_ai_enhancement.inference.tiling import TileConfig, tiled_inference


class FirstFour(nn.Module):
    """用于隔离分块逻辑的逐像素四通道参考模型。"""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """返回不依赖邻域的确定性残差，便于精确比较接缝。"""

        return value[:, :4] * 0.1


def test_tiling_matches_full_frame_for_pointwise_model() -> None:
    """逐像素模型的分块输出应与整帧输出逐元素一致。"""

    value = torch.rand(1, 16, 70, 75)
    value[:, 15] = 1
    expected = FirstFour()(value)
    result = tiled_inference(FirstFour(), value, TileConfig(size=32, halo=8, multiple=16))
    torch.testing.assert_close(result.residual, expected, atol=1e-6, rtol=1e-6)
    assert result.tile_count > 1
