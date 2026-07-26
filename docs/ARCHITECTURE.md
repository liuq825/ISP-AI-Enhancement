# 架构与关键决策

## 运行图

模型接收一张 canonical packed RAW 和静态上下文。HDR/MFNR 路径在融合后只调用
一次模型；AI 不负责重新配准，只根据 confidence 和 motion/ghost 条件执行保守
残差恢复。

```text
RAW / fused RAW
  -> Sensor normalization and canonical packing
  -> 16-channel context
  -> NAFNet RAW residual
  -> FP16/FP32 clip(raw + residual)
  -> inverse calibration / demosaic / color ISP
```

## 模型

- Teacher：`width=64`，只在训练侧提供输出和特征监督。
- Student P0：`width=32`，主通道与四级拓扑固定。
- P1/P2/P3：只减少 NAFBlock 内部的两个成对扩展分支，并物理重建卷积。
- 输出：四通道 residual；模型不在图内执行最终 clip，便于首尾混合精度控制。

## 对原方案的数字核验与修正

代码统计的 P0 精确参数为 `14,348,516`，因此原方案的约 14.347M 是准确近似。
需要修正的是统一 Middle=400 的结构模板：它会使总剪枝明显超过 15%。仓库参考
配置改为按块 `[416, 416, 432, 432]`，并保留 Encoder/Decoder 的 16 对齐方案；
精确结果为 `12,176,868` 参数、`15.135%` 物理剪枝。最终比率由
`isp-ai pruning-summary` 输出，任何文档常数都不能替代模型图统计。

参考配置不是最终通道索引。生产剪枝必须使用真实训练权重和分域敏感度，确保
SimpleGate 两半、Depthwise、SCA 和前后 1×1 的索引同步。

## 部署后端决策

麒麟端侧存在两条不同交付路线：

1. 首选：产品团队提供的 HiAI CANN DDK，生成并加载离线 `.om`。
2. 备用：MindSpore Lite 将 ONNX 转为 `.ms`，用于 API 和功能验证。

两条路线的算子、量化和性能不能互相推定。`.ms` 跑通不代表 `.om` 已达标；服务器
昇腾 ATC 跑通也不代表移动端目标固件支持同一算子。
