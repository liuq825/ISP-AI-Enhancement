# 数据集目录与使用策略

机器可读清单位于 `resources/datasets.yaml`，检查日期为 2026-07-26。

## 已核验来源

- SIDD：官方页面说明数据集与关联代码采用 MIT License，包含约 30,000 张真实噪声
  图像及 RAW-RGB/元数据，可用作管线和研究基线。
- SID：官方仓库采用 MIT，但数据文件权利范围仍需单独确认；默认不进入生产权重。
- DND：官方许可明确限定非商业用途并禁止再分发，只能隔离做研究评测。
- ELD：公开材料中的数据许可不够明确，默认禁用，直到获得书面许可。

数据不会提交进 Git。下载、人工注册、许可接受和法务审批必须由具名人员完成并保存
许可快照。对商用模型，公开集最多是预研补充，生产主数据必须来自拥有或已获许可的
目标 Sensor 采集。

## 为什么公开集不能替代产品数据

公开集无法覆盖目标设备的 CFA、黑电平、模拟/数字增益链、行列噪声、温度漂移、
镜头阴影、坏点、HDR/MFNR 融合残余以及实际固件版本。只在公开集上获得高 PSNR
不能证明可在产品上商业部署。

## Manifest 约束

每条 JSONL 记录至少包含：

`sample_id, input_path, target_path, split, sensor_id, mode, session_id,
scene_id, iso_bucket, metadata`

切分以 `session_id + scene_id` 为最小隔离组。相邻帧、同一 burst 或同一标定序列
不得跨 train/val/test。`validate-manifest` 会拒绝重复 ID、缺失文件和组级泄漏。

## 烟雾数据

`make-smoke-data` 生成完全程序化、无外部版权依赖的 packed RAW。它包含
shot/read、行、列和黑电平漂移噪声，只用于测试代码和 CI，不能用于画质结论。
