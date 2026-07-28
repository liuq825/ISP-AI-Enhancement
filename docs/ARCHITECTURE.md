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

- Teacher：`width=64`，只在训练侧提供输出、feature 和空间 attention 监督。
- Student P0：`width=32`、`encoder_blocks=[2,2,6,8]`；块数由 YAML 配置，不在
  训练、剪枝或导出代码中写死。相比旧 `[2,2,4,8]`，enc3 增加两个中尺度 block。
- P1/P2/P3：只减少 NAFBlock 内部的两个成对扩展分支；按结构敏感性分配预算并物理
  重建卷积，而不是让所有 block 统一承担 15%。
- 输出：四通道 residual；模型不在图内执行最终 clip，便于首尾混合精度控制。

## 对原方案的数字核验与修正

旧 `[2,2,4,8]` P0 的 `14,348,516` 参数只是上一轮基线。升级 enc3 后，当前 P0
精确参数为 `14,586,340`。结构感知目标完整保留 enc1/enc2/dec3/dec4，保护
enc3/enc4 首尾块，并把更多预算放到 Middle 和深层 Decoder；目标为
`12,405,108` 参数、`14.953936%` 物理剪枝。最终比率及逐 stage 隐藏通道保留率
由 `isp-ai pruning-summary` 输出，文档常数不能替代模型图统计。

参考结构不是最终通道索引。正式剪枝必须在已收敛 P0 上按 Sensor×模式×ISO 做
逐 stage/逐 block 消融，并使用真实权重重要性选择通道；任何域超出画质预算就回退
该 block 的压缩。SimpleGate 两半、Depthwise、SCA 和前后 1×1 的索引必须同步。

当前默认由 Torch-Pruning DepGraph 执行上述联动删除，手工重建后端作为独立基线。
实现约束、命令和已知兼容性问题见 `PRUNING.md`。

## 联合蒸馏

Student 训练不再只拟合 Teacher 输出。`configs/distill.yaml` 同时使用：

1. 低权重 Teacher 输出残差约束；
2. 1×1 投影后的多尺度 feature Smooth-L1；
3. 通道均方能量、逐样本 L2 归一化后的空间 attention MSE。

feature 传递表征值，attention 强调边缘、纹理和噪声残差集中区域；两类适配器只在
训练期存在，不进入 ONNX/OM。具体层位、权重和恢复约束见 `DISTILLATION.md`。

## 部署后端决策

麒麟端侧存在两条不同交付路线：

1. 首选：产品团队提供的 HiAI CANN DDK，生成并加载离线 `.om`。
2. 备用：MindSpore Lite 将 ONNX 转为 `.ms`，用于 API 和功能验证。

两条路线的算子、量化和性能不能互相推定。`.ms` 跑通不代表 `.om` 已达标；服务器
昇腾 ATC 跑通也不代表移动端目标固件支持同一算子。
