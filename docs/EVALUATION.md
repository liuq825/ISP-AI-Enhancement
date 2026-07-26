# RAW 质量评测规范

## 统一口径

`isp-ai evaluate` 是公开集基线、P0、剪枝模型和 QAT 候选的统一离线评测入口。每个样本
先在有效区域内计算四通道 packed RAW MSE、PSNR 和高斯窗 SSIM，再对样本做等权平均。
不能先把整个 batch 的像素混成一个 MSE，也不能对 batch 均值再次等权平均，否则最后一个
不足 batch 的批次会得到错误权重。

报告同时输出：

- 全集的 noisy 输入 PSNR；
- 全集的 noisy packed RAW SSIM；
- 提供模型时的增强 PSNR/SSIM 与相对增益；
- 按 `sensor_id` 和 `iso_bucket` 的分桶结果；
- Manifest、模型配置和 checkpoint 的 SHA256；
- 设备、样本数、数据用途与评测耗时。

## SIDD 验证基线

```powershell
.\.venv\Scripts\isp-ai.exe evaluate `
  --manifest data/sidd_validation/manifest.jsonl `
  --split test `
  --context-config configs/context.yaml `
  --catalog resources/datasets.yaml `
  --purpose commercial_grade `
  --output outputs/sidd_validation_noisy_baseline.json
```

加入模型：

```powershell
.\.venv\Scripts\isp-ai.exe evaluate `
  --manifest data/sidd_validation/manifest.jsonl `
  --split test `
  --context-config configs/context.yaml `
  --catalog resources/datasets.yaml `
  --purpose commercial_grade `
  --model-config configs/model_student.yaml `
  --checkpoint checkpoints/student_p0.pt `
  --device cuda `
  --batch-size 16 `
  --output outputs/student_p0_sidd_validation.json
```

## 放行边界

公开 SIDD 的高 PSNR 只证明公开域去噪能力，不能代替目标 Sensor 的坏点、LSC、融合残余、
温漂、色偏和实机主观画质。商用品质候选还必须在冻结的目标 Sensor Golden Set 上按
Sensor×模式×ISO 分桶，并执行纹理、色彩、鬼影、热态性能和回退门禁。

内部 SSIM 在四个 canonical packed RAW 通道上分别计算 11×11 高斯窗统计；mask 只有在
整个窗口有效时才纳入平均。它用于模型版本间稳定对照，不宣称与 SIDD 服务器在 Bayer
mosaic 上的实现逐位相同，官方成绩仍以官方评测为准。
