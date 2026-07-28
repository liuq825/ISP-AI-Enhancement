# 本机 QAT 全链路运行记录

运行日期：2026-07-28。该记录证明本机 CPU 上的工程链路可执行，不代表真实 RAW 画质、
麒麟 9000 NPU 性能，或最终部署放行。

## 已执行命令

```powershell
.\.venv\Scripts\isp-ai.exe make-smoke-data --output data/local_phase2_smoke --samples 16
.\.venv\Scripts\isp-ai.exe validate-manifest --manifest data/local_phase2_smoke/manifest.jsonl
.\.venv\Scripts\isp-ai.exe train --config configs/train_local_fp32_smoke.yaml
.\.venv\Scripts\isp-ai.exe train --config configs/train_local_qat_smoke.yaml
.\.venv\Scripts\isp-ai.exe export-onnx `
  --config configs/model_smoke.yaml `
  --checkpoint runs/local_qat_smoke/best.pt `
  --qat-config configs/qat.yaml `
  --export-config configs/export_local_qat_smoke.yaml `
  --output artifacts/local_qat_smoke_64.onnx
.\.venv\Scripts\isp-ai.exe audit-onnx --model artifacts/local_qat_smoke_64.onnx
```

## 结果

| 项目 | 结果 |
|---|---|
| 合成数据与 Manifest | 16 个样本，校验通过 |
| FP32 基线 | 一轮训练完成，`runs/local_fp32_smoke/best.pt` 可加载 |
| QAT | 一轮训练完成，62 个卷积进入伪量化，2 个首尾卷积保留高精度 |
| QAT 权重 SHA256 | `38e0ab51ece48269dfd1a1764e0224db5038fdc8bd36658bec071021bf855ee3` |
| QAT ONNX SHA256 | `c934ea82b31d7f20d62880457269239e13b1754a3329ed2dae6c37b8da19dc3c` |
| ONNX 输入/输出 | 静态 `1×16×64×64` → `1×4×64×64` |
| ONNX Checker 与 ORT 对照 | 通过；最大绝对误差 `3.28e-7` |
| Q/DQ 节点 | `QuantizeLinear=124`，`DequantizeLinear=124` |
| 模拟 INT8 权重比例 | `99.4964%` |

## 本机 QAT 导出容差

`configs/export_local_qat_smoke.yaml` 只服务本地极小模型与 `64×64` 输入。它记录了独立的
误差容差，原因是对同一烟雾 QAT 权重以 `512×512` 导出时，PyTorch/ORT 的舍入边界曾出现
`2.156e-3` 最大绝对误差。该现象已记录为本地兼容性风险，不能通过放宽正式模型的
`configs/export_onnx.yaml` 容差来掩盖。

正式 Student 的导出仍必须使用严格配置、真实收敛 QAT 权重和目标 HiAI/CANN DDK；最终 OM、
算子落点、热态时延、功耗与稳定性证据缺失时，模型 manifest 必须保持
`UNVERIFIED_UNTIL_TARGET_HIAI_CANN_PROFILING`。
