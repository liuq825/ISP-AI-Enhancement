import torch
from torch import nn

from isp_ai_enhancement.inference.tiling import TileConfig, tiled_inference


class FirstFour(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :4] * 0.1


def test_tiling_matches_full_frame_for_pointwise_model() -> None:
    value = torch.rand(1, 16, 70, 75)
    value[:, 15] = 1
    expected = FirstFour()(value)
    result = tiled_inference(FirstFour(), value, TileConfig(size=32, halo=8, multiple=16))
    torch.testing.assert_close(result.residual, expected, atol=1e-6, rtol=1e-6)
    assert result.tile_count > 1
