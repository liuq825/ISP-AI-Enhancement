# 麒麟 9000 部署路线

## 已确认事实

华为 HiAI 是移动端 NPU 能力入口；官方 Foundation 示例在麒麟 810/990 及以后设备
加载离线 `.om`。当前华为 ML Kit 文档也提供 MindSpore Lite 自定义模型路线，
将主流格式转换为 `.ms` 在端侧运行。

## 本项目路线

### A. HiAI CANN/OM（量产目标）

1. 用固定 `N=1, C=16, H/W=Tile` 导出 ONNX。
2. 在目标产品团队提供的 DDK/转换器上生成 OM；转换命令和版本写入 manifest。
3. 审计每个节点的 NPU/CPU 落点、Cast/Requant、首尾精度和实际计算量。
4. 在目标机型运行 384/512/640 Tile A/B，测热态 P50/P90/P99、峰值内存和功耗。
5. 锁定 DDK、系统、驱动、固件、OM 和 Sensor 校准包的兼容矩阵。

仓库不固定一个臆测的 `atc` 或 `omg` 参数集。不同 HiAI CANN DDK 的转换器名称、
ONNX 支持和参数会变化；`deploy/hiai/invoke_converter.ps1` 要求产品团队提供并
版本化真实参数文件。

进入 DDK 转换前，ONNX 必须具有通过
`resources/model_manifest.schema.json` 约束的伴生清单；清单包含 raw16-v1
通道顺序、源文件哈希、ORT 对照和算子审计。其状态仍保持
`UNVERIFIED_UNTIL_TARGET_HIAI_CANN_PROFILING`，直到真机 Gate 完成。

### B. MindSpore Lite（备用验证）

用于验证宿主 API、Tile、回退和 Android/HarmonyOS 集成。它不是 OM 性能结论的
替代。转换后仍需在具体设备和 delegate/NPU 后端检查实际落点。

## 高风险算子

NAFNet 的关键路径包含 channel-wise LayerNorm、逐元素乘法、AdaptiveAvgPool、
Depthwise Conv 和 PixelShuffle。仓库将 LayerNorm 表达为基础算子以提高可转换性，
但是否融合、是否回退以及 INT8 精度只能由目标转换日志和 Profiler 判定。

## Tile

首选候选为 packed `512×512, halo=48`，但它只是起点。重叠区使用二维 Hann 权重；
Pad 区同步更新 valid mask。Tile 只降低峰值激活，不降低总 MACs。

## 产品级回退

模型缺失、校准/契约版本不匹配、输入异常、NPU 超时、温控触发或输出 NaN 时，
必须 100% 回退传统 ISP/已认证 Lite 路径，并记录稳定 reason code；AI 故障不得
阻断拍照出图。
