# 数据集目录与使用策略

机器可读清单位于 `resources/datasets.yaml`，检查日期为 2026-07-26。

## 已核验来源

- SIDD：官方页面说明数据集与关联代码采用 MIT License，包含约 30,000 张真实噪声
  图像及 RAW-RGB/元数据，可用作管线和研究基线。
- SID：官方仓库采用 MIT，但数据文件权利范围仍需单独确认；默认不自动导入。
- DND：官方许可明确限定非商业用途并禁止再分发，只能隔离做研究评测。
- ELD：公开材料中的数据许可不够明确，默认禁用，直到获得书面许可。

当前交付是“达到商用级技术指标的非商业工程验证模型”，不是直接商业发布，因此
商业版权/IP 清关不作为本项目完成前置条件。数据仍不会提交进 Git，且必须遵守来源
自身的用途、注册和禁止再分发条款。未来若直接商业发布，应重新做 IP 审计，并优先
使用目标 Sensor 自有数据重训。

## SIDD 导入

官方 SIDD RAW-RGB `.MAT` 实际是黑电平已扣除、归一化到 `[0,1]` 的二维 Bayer
mosaic，并不是本项目所需的四通道 canonical RAW。`import-sidd` 根据官方 CFA 表
执行以下映射：

`GP=BGGR, IP=RGGB, S6=GRBG, N6=BGGR, G4=BGGR`

转换结果固定为 `[R, Gr, Gb, B]`，同时记录原始文件 SHA256、ISO、曝光分母、CCT、
CFA 与逐场景 NLF。数据切分使用三位 `scene_id` 分组；同一物理场景即使由不同相机
或 ISO 拍摄，也不会跨越 train/val/test。

`noise_sigma` 根据官方 NLF 在线性 18% 参考灰处计算
`sqrt(mean(beta1*0.18 + beta2))`。它只是 16 通道上下文的可复现标量摘要，完整
六个 NLF 系数仍保存在 Manifest 元数据中。

## 为什么公开集不能替代产品数据

公开集无法覆盖目标设备的 CFA、黑电平、模拟/数字增益链、行列噪声、温度漂移、
镜头阴影、坏点、HDR/MFNR 融合残余以及实际固件版本。只在公开集上获得高 PSNR
不能证明已经达到目标麒麟 9000 设备上的商用级画质与性能指标。

## Manifest 约束

每条 JSONL 记录至少包含：

`sample_id, dataset_id, input_path, target_path, split, sensor_id, mode,
session_id, scene_id, iso_bucket, metadata`

切分以 `session_id + scene_id` 为最小隔离组。相邻帧、同一 burst 或同一标定序列
不得跨 train/val/test。`validate-manifest` 会拒绝重复 ID、缺失文件和组级泄漏。

## 烟雾数据

`make-smoke-data` 生成完全程序化、无外部版权依赖的 packed RAW。它包含
shot/read、行、列和黑电平漂移噪声，只用于测试代码和 CI，不能用于画质结论。
