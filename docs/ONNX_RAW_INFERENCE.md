# ONNX RAW 推理验证脚本

脚本 [`scripts/verify_onnx_raw_pipeline.py`](../scripts/verify_onnx_raw_pipeline.py) 验证
`raw16-v1` ABI：输入为单帧 Sensor Bayer，或 HDR/MFNR 上游已经融合好的四通道 packed RAW；
脚本构建 16 通道上下文、调用 ONNX、加回四通道残差，并输出可继续进入 Demosaic/传统 ISP 的
归一化 RAW。

## 输入约定

| 输入类型 | 参数 | 形状 | 说明 |
|---|---|---|---|
| 单帧 Sensor RAW | `--raw-layout mosaic` | `H×W` | 原始 Bayer 马赛克；脚本按 `--cfa-pattern` 规范为 `[R,Gr,Gb,B]`。 |
| HDR/MFNR 融合 RAW | `--raw-layout packed` | `4×H×W` 或 `H×W×4` | 必须已按 canonical `[R,Gr,Gb,B]` 排列和融合完成。 |
| 融合置信度 | `--fusion-confidence-npy` | `H×W` 等价形状 | 可选，填入第 7 通道；HDR/MFNR 不提供时默认 0，不能伪装为可信融合。 |
| 运动鬼影 | `--motion-ghost-npy` | `H×W` 等价形状 | 可选，填入第 8 通道。 |

`--black-level` 与 `--white-level` 必须使用真实 Sensor 元数据。输出仍在 `[0,1]`；传统 ISP
若要求 ADC 码值，应按同一黑白电平逆变换。输出 `mosaic` 时分辨率为 packed RAW 的两倍，
可直接接 Bayer Demosaic；输出 `packed` 时保留 canonical 四通道形式。

## 单帧 Sensor RAW 验证

```powershell
.\.venv\Scripts\python.exe scripts\verify_onnx_raw_pipeline.py `
  --model artifacts\local_full_pipeline_qat_64.onnx `
  --raw-npy input\sensor_raw_rggb.npy `
  --raw-layout mosaic --cfa-pattern RGGB `
  --sensor-id smoke_sensor --mode single `
  --black-level 64 --white-level 4095 `
  --noise-sigma 0.02 --exposure-ratio 1.0 --wb-rg 1.8 --wb-bg 1.5 `
  --output output\enhanced_sensor_raw.npy --output-layout mosaic `
  --report output\sensor_raw_report.json
```

## HDR/MFNR 融合 RAW 验证

```powershell
.\.venv\Scripts\python.exe scripts\verify_onnx_raw_pipeline.py `
  --model artifacts\local_full_pipeline_qat_64.onnx `
  --raw-npy input\fused_packed_raw.npy --raw-layout packed `
  --sensor-id smoke_sensor --mode mfnr `
  --black-level 0 --white-level 1 `
  --fusion-confidence-npy input\fusion_confidence.npy `
  --motion-ghost-npy input\motion_ghost.npy `
  --output output\enhanced_mfnr_raw.npy --output-layout mosaic `
  --report output\mfnr_report.json
```

## 本机已验证结果

使用 `artifacts/local_full_pipeline_qat_64.onnx` 已实际验证：

- 合成单帧 Bayer：`1×16×64×64` 输入，输出 `128×128` Bayer mosaic，数值范围 `[0,1]`；
- 已融合 packed RAW 的 MFNR 模式：同一静态 ONNX 输入，输出 `128×128` Bayer mosaic，数值范围 `[0,1]`。

该验证证明 ONNX Runtime、16 通道 ABI 与传统 ISP 前 RAW 输出可工作。它不证明目标 Sensor
的标定正确性，也不证明麒麟 9000 的 OM、NPU 算子、时延、功耗或热稳定性已通过。
