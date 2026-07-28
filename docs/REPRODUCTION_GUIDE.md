# 项目复现操作指南

本指南给新开发者提供最短可执行路径。命令以 Windows PowerShell 为准，均在仓库根目录
运行。公开数据可复现“商用级技术目标研发基线”，不等于已经完成目标 Sensor 授权、
麒麟 9000 真机或正式商业发布。

## 1. 环境安装

```powershell
git clone https://github.com/liuq825/ISP-AI-Enhancement.git
Set-Location ISP-AI-Enhancement
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
```

基础验证：

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q
```

当前预期为 `71 passed`。4 条旧 TorchScript ONNX 导出器弃用告警已记录，不是测试失败。

## 2. 本机 CPU 快速闭环

当前开发机无 CUDA，先运行合成 RAW：

```powershell
.\.venv\Scripts\isp-ai.exe make-smoke-data --output data/smoke --samples 16
.\.venv\Scripts\isp-ai.exe validate-manifest --manifest data/smoke/manifest.jsonl
.\.venv\Scripts\isp-ai.exe train --config configs/train_smoke.yaml
```

预期生成 `runs/smoke/epoch_0001.pt` 和 `runs/smoke/history.jsonl`。该权重只验证代码，
没有画质意义。本机资源与任务边界见 `docs/LOCAL_DEVELOPMENT_PROFILE.md`。

若要从零演练 FP32→QAT→ONNX 的本地闭环，执行：

```powershell
.\.venv\Scripts\isp-ai.exe make-smoke-data --output data/local_phase2_smoke --samples 16
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

结果和边界见 `docs/LOCAL_QAT_SMOKE_RUN.md`；这不是正式麒麟候选模型。
仓库同时版本化该本机闭环的 checkpoint、训练 history、ONNX 与 manifest；克隆仓库后可直接
审计这些交付物，也可用上述命令在本机重新生成。数据集、虚拟环境和缓存不纳入 Git。

使用 ONNX 对单帧 Sensor RAW 或 HDR/MFNR 融合 RAW 做 Demosaic 前验证的命令见
`docs/ONNX_RAW_INFERENCE.md`。

## 3. 获取 SIDD Medium 规模 RAW

生成固定 160 场景×2 帧配置：

```powershell
.\.venv\Scripts\isp-ai.exe build-sidd-range-config `
  --output resources/sidd_medium_range.yaml --frames 10 20
```

获取数据；CodaLab 持续故障时才加 `--prefer-fallback`：

```powershell
.\.venv\Scripts\isp-ai.exe fetch-sidd-subset `
  --config resources/sidd_medium_range.yaml `
  --output datasets/SIDD_Medium_Range `
  --progress-file outputs/sidd_medium_download.log `
  --max-attempts 12 --retry-backoff-seconds 10
```

状态和全量 MAT 审计：

```powershell
.\.venv\Scripts\isp-ai.exe sidd-fetch-status `
  --config resources/sidd_medium_range.yaml `
  --output datasets/SIDD_Medium_Range
.\.venv\Scripts\isp-ai.exe audit-sidd-subset `
  --config resources/sidd_medium_range.yaml `
  --source datasets/SIDD_Medium_Range `
  --output resources/sidd_medium_receipt.yaml
```

预期：160 场景、320 对、640 MAT、无 `.partial`。获取回执 SHA256 为
`cd1b74d6cb54a618d5b7dd792af94e27a2495f25d353e5d7e3b62c20b89c4aa2`。

## 4. 生成训练 patch 并审计

先从 [SIDD 官方网站](https://abdokamel.github.io/sidd/)获取验证/支持文件，并将噪声级函数
文件放到 `datasets/SIDD_Blocks/noise_level_functions.csv`。其 SHA256 应为：

```text
e48a235ae008bf6076402633df0dd1dc1fa3b6afce1ee0558a05ed358593d6b9
```

哈希不一致时不要继续正式训练，应先确认下载来源、文件版本，以及是否发生了解压或文本
格式转换。随后生成训练 patch：

```powershell
.\.venv\Scripts\isp-ai.exe import-sidd `
  --source datasets/SIDD_Medium_Range `
  --output data/sidd_training `
  --nlf-csv datasets/SIDD_Blocks/noise_level_functions.csv `
  --patch-size 256 --patches-per-pair 16 --patch-seed 20260727
.\.venv\Scripts\isp-ai.exe validate-manifest `
  --manifest data/sidd_training/manifest.jsonl
.\.venv\Scripts\isp-ai.exe audit-sidd-import `
  --manifest data/sidd_training/manifest.jsonl `
  --training-config configs/train_student_public_baseline.yaml `
  --acquisition-receipt resources/sidd_medium_receipt.yaml `
  --nlf-csv datasets/SIDD_Blocks/noise_level_functions.csv `
  --output resources/sidd_medium_import_receipt.yaml
```

预期：

- 5,120 条 Manifest、10,240 个 NPZ；
- train/val/test patch 为 3,296/96/1,728；
- train/val/test 独立源配对为 206/6/108；
- 数组内容 SHA256 为
  `d9dba6d6ac422d8cc0cf20f9b69f9bd21e019f4c5b5925d543b360d47dae3876`；
- 当前导入回执 SHA256 为
  `d26597ad6ade7f3330a9714bd563037966277e0776fdf792e16b120bfa6ba183`。

## 5. CUDA 正式训练

以下步骤不能在当前 CPU 本机冒充完成。先记录训练机 GPU、驱动、CUDA、显存和依赖
锁定结果，再按顺序执行。

### 5.1 Teacher

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs/train_teacher.yaml
```

预期最佳权重：`runs/teacher/best.pt`。

### 5.2 `[2,2,6,8]` Student 联合蒸馏

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs/distill.yaml
```

预期最佳权重：`runs/student_feature_attention_distill/best.pt`。日志必须同时包含
`loss_teacher_output`、`loss_teacher_feature`、`loss_teacher_attention`。

### 5.3 结构感知约 15% 剪枝

先复核静态结构：

```powershell
.\.venv\Scripts\isp-ai.exe pruning-summary `
  --source configs/model_student.yaml `
  --target configs/model_student_structaware15.yaml `
  --backend torch-pruning
```

预期 `14,586,340 → 12,405,108`，物理剪枝率 `14.953936%`。再剪已收敛权重：

```powershell
.\.venv\Scripts\isp-ai.exe prune-checkpoint `
  --source-config configs/model_student.yaml `
  --source-checkpoint runs/student_feature_attention_distill/best.pt `
  --target-config configs/model_student_structaware15.yaml `
  --output checkpoints/student_structaware15.pt `
  --backend torch-pruning
.\.venv\Scripts\isp-ai.exe train --config configs/train_structaware15.yaml
```

必须用真实验证集完成逐 block/逐域敏感度回退；静态参考宽度不等于最终最优结构。

### 5.4 QAT

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs/train_qat.yaml
```

QAT 从 `runs/student_structaware15_finetune/best.pt` 开始，不得直接使用刚剪枝权重。

## 6. ONNX 与麒麟部署

```powershell
.\.venv\Scripts\isp-ai.exe export-onnx `
  --config configs/model_student_structaware15.yaml `
  --checkpoint runs/student_structaware15_qat/best.pt `
  --qat-config configs/qat.yaml `
  --output artifacts/student_structaware15_qat_512.onnx
.\.venv\Scripts\isp-ai.exe audit-onnx `
  --model artifacts/student_structaware15_qat_512.onnx
```

HiAI/CANN DDK 由产品团队提供后：

```powershell
New-Item -ItemType Directory -Force evidence | Out-Null
Copy-Item deploy\hiai\converter.args.example `
  evidence\hiai_kirin9000_DDK_VERSION.args
# 编辑上述参数文件，把 ONNX/输出路径和目标 SoC 参数替换为产品团队确认值。
.\deploy\hiai\invoke_converter.ps1 `
  -ConverterPath 'C:\Vendor\HiAI\converter.exe' `
  -ArgumentsFile evidence\hiai_kirin9000_DDK_VERSION.args
```

必须把真实 DDK/转换器版本、参数文件、ONNX/OM SHA256 和 profiler 结果写入模型
manifest。没有目标 DDK 时只能验证 ONNX/MindSpore Lite 备用路径，不能宣称完成 OM。

## 7. 发布 Gate

复制 `resources/release_evidence.example.yaml` 为实际证据文件，填入目标 Sensor、
模式、ISO、画质、OM、热态性能、内存、稳定性和签字信息，然后执行：

```powershell
.\.venv\Scripts\isp-ai.exe check-release `
  --evidence evidence/kirin9000_release.yaml `
  --output evidence/kirin9000_release.report.json
```

只有总体状态 `PASS` 才可放行；缺设备、DDK 或证据时应保持 `BLOCKED`，真实越界为
`FAIL`。完整阈值见 `docs/RELEASE_GATES.md`。

## 8. 复现记录要求

每次实验至少保存：

- Git commit、Python/PyTorch/CUDA/驱动和 GPU 型号；
- 配置文件及 SHA256、数据回执 SHA256；
- checkpoint、ONNX、OM 及伴生 manifest；
- history、逐 Sensor×模式×ISO 指标和失败样例；
- 实际命令、耗时、峰值显存/内存、异常与解决方式。

遇到问题先查 `docs/ENGINEERING_LOG.md`，新增坑按“现象→根因→处理→防复发”补充，确保
后续人员不依赖口头信息。
