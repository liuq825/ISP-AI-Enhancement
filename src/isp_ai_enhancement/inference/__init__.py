"""导出端侧大图分块推理配置、结果和执行函数。"""

from .tiling import TileConfig, TiledResult, tiled_inference

__all__ = ["TileConfig", "TiledResult", "tiled_inference"]
