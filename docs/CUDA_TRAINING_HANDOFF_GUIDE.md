# CUDA 训练机独立操作交接指南

本指南供没有 Codex 的操作人员完成剩余研发工作。目标是得到真实 SIDD 数据训练的
`[2,2,6,8]` Student、结构感知约 15% 剪枝、QAT ONNX，以及可交付目标 HiAI/CANN DDK
转换的证据包。所有命令在仓库根目录执行，示例使用 Windows PowerShell。

> 训练完成不等于麒麟 9000 已部署完成。只有目标 DDK 生成 OM、真机 profiler 和 Release
> Gate 证据齐全后，才可声明端侧部署通过。

## 0. 获取完全一致的代码

训练机必须包含本地最新提交 `774ab5b` 或之后的提交。若 GitHub 已同步：

```powershell
git clone https://github.com/liuq825/ISP-AI-Enhancement.git
Set-Location ISP-AI-Enhancement
git log -1 --oneline
```

若 GitHub 尚未同步，在原开发机仓库执行并通过 U 盘/内网传输 bundle：

```powershell
git bundle create ISP-AI-Enhancement-cuda-handoff.bundle main
```

在训练机解包：

```powershell
git clone X:\ISP-AI-Enhancement-cuda-handoff.bundle ISP-AI-Enhancement
Set-Location ISP-AI-Enhancement
git remote add origin https://github.com/liuq825/ISP-AI-Enhancement.git
git log -1 --oneline
```

检查代码、配置、脚本均已存在：

```powershell
Test-Path configs\train_teacher.yaml
Test-Path configs\distill.yaml
Test-Path configs\train_structaware15.yaml
Test-Path configs\train_qat.yaml
Test-Path scripts\verify_onnx_raw_pipeline.py
```

## 1. CUDA 环境安装与验收

建议使用 NVIDIA CUDA GPU、至少 16 GiB 显存、至少 32 GiB 系统内存和 100 GiB 可用磁盘。
显存较小也可从 `batch_size: 1` 开始，但必须降低 `crop_size` 或增加梯度累积，且要把实际
修改记录到新的运行配置和实验日志，不能改写历史配置。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# 先按 https://pytorch.org/get-started/locally/ 安装与驱动匹配的 CUDA 版 PyTorch。
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

最后一条 Python 命令必须输出 `True` 和实际 GPU 名称；否则不得启动正式 CUDA 训练。
记录 GPU、显存、驱动、PyTorch、CUDA、操作系统、Git commit 与依赖版本：

```powershell
nvidia-smi | Tee-Object -FilePath evidence\cuda_environment.txt
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding UTF8 evidence\pip_freeze.txt
git rev-parse HEAD | Set-Content -Encoding UTF8 evidence\git_commit.txt
```

## 2. 获取或迁移数据集

Git 不包含数据集。最快路径是从原开发机复制整个 `data\sidd_training\` 目录，并同时复制：

- `datasets\SIDD_Medium_Range\`（用于重新审计）；
- `datasets\SIDD_Blocks\noise_level_functions.csv`；
- `resources\sidd_medium_receipt.yaml` 与 `resources\sidd_medium_import_receipt.yaml`。

若不能迁移，按 [SIDD Medium 获取说明](SIDD_MEDIUM_ACQUISITION.md) 和
[复现指南](REPRODUCTION_GUIDE.md) 从零下载、导入、审计。训练前必须运行：

```powershell
.\.venv\Scripts\isp-ai.exe validate-manifest --manifest data\sidd_training\manifest.jsonl
.\.venv\Scripts\isp-ai.exe audit-sidd-import `
  --manifest data\sidd_training\manifest.jsonl `
  --training-config configs\train_student_public_baseline.yaml `
  --acquisition-receipt resources\sidd_medium_receipt.yaml `
  --nlf-csv datasets\SIDD_Blocks\noise_level_functions.csv `
  --output evidence\sidd_medium_import_receipt.cuda.yaml
```

只有 Manifest 和审计都通过，才可开始训练。不要用 `data/smoke` 或小样本代替正式训练。

## 3. 训练 Teacher

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs\train_teacher.yaml `
  2>&1 | Tee-Object -FilePath runs\teacher\console.log
```

期望产物：`runs\teacher\best.pt`、`epoch_*.pt`、`history.jsonl`。每次训练结束后评估：

```powershell
.\.venv\Scripts\isp-ai.exe evaluate `
  --manifest data\sidd_training\manifest.jsonl --split val `
  --model-config configs\model_teacher.yaml --checkpoint runs\teacher\best.pt `
  --device cuda --output evidence\teacher_val.json
```

中断恢复时，复制原配置为新文件，在其中增加 `resume_checkpoint: runs/teacher/epoch_XXXX.pt`，
保留相同 `output_dir`，并把新配置和恢复命令写入实验记录。不得把 `best.pt` 当作恢复点。

## 4. 训练 Feature + Attention Student

确认 Teacher 已存在后执行：

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs\distill.yaml `
  2>&1 | Tee-Object -FilePath runs\student_feature_attention_distill\console.log
```

检查 `history.jsonl` 同时存在 `loss_teacher_output`、`loss_teacher_feature`、
`loss_teacher_attention`；缺任一项即表示不是完整联合蒸馏。评估：

```powershell
.\.venv\Scripts\isp-ai.exe evaluate `
  --manifest data\sidd_training\manifest.jsonl --split val `
  --model-config configs\model_student.yaml `
  --checkpoint runs\student_feature_attention_distill\best.pt `
  --device cuda --output evidence\student_distill_val.json
```

## 5. 结构感知约 15% 剪枝与恢复微调

先复核拓扑，预期 Student 从 `14,586,340` 到 `12,405,108` 参数，约 `14.953936%`：

```powershell
.\.venv\Scripts\isp-ai.exe pruning-summary `
  --source configs\model_student.yaml `
  --target configs\model_student_structaware15.yaml `
  --backend torch-pruning | Tee-Object evidence\pruning_summary.json
```

从已收敛 Student 物理剪枝：

```powershell
.\.venv\Scripts\isp-ai.exe prune-checkpoint `
  --source-config configs\model_student.yaml `
  --source-checkpoint runs\student_feature_attention_distill\best.pt `
  --target-config configs\model_student_structaware15.yaml `
  --output checkpoints\student_structaware15.pt `
  --backend torch-pruning | Tee-Object evidence\pruned_checkpoint_manifest.json
.\.venv\Scripts\isp-ai.exe train --config configs\train_structaware15.yaml `
  2>&1 | Tee-Object -FilePath runs\student_structaware15_finetune\console.log
```

`train_structaware15.yaml` 已指向 `data/sidd_training/manifest.jsonl` 并复用正式数据门槛。
禁止跳过恢复微调直接 QAT。

## 6. QAT、导出和 ONNX 审计

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs\train_qat.yaml `
  2>&1 | Tee-Object -FilePath runs\student_structaware15_qat\console.log
.\.venv\Scripts\isp-ai.exe evaluate `
  --manifest data\sidd_training\manifest.jsonl --split val `
  --model-config configs\model_student_structaware15.yaml `
  --checkpoint runs\student_structaware15_qat\best.pt `
  --device cuda --output evidence\student_qat_val.json
.\.venv\Scripts\isp-ai.exe export-onnx `
  --config configs\model_student_structaware15.yaml `
  --checkpoint runs\student_structaware15_qat\best.pt `
  --qat-config configs\qat.yaml `
  --output artifacts\student_structaware15_qat_512.onnx
.\.venv\Scripts\isp-ai.exe audit-onnx `
  --model artifacts\student_structaware15_qat_512.onnx `
  | Tee-Object evidence\student_qat_onnx_audit.json
```

导出会生成 `.manifest.json`。确认其包含静态 `1×16×512×512` 输入、Q/DQ 节点、Checker/ORT
数值结果和 `UNVERIFIED_UNTIL_TARGET_HIAI_CANN_PROFILING` 状态。该状态在拿到真机证据前
必须保留，不能手动改为已验证。

## 7. RAW 输入到 Demosaic 前输出验证

使用 [ONNX RAW 推理说明](ONNX_RAW_INFERENCE.md) 中的脚本。正式 512 packed RAW ONNX
要求上游提供 `1024×1024` Bayer mosaic，或使用 Tile/Padding 后再调用：

```powershell
.\.venv\Scripts\python.exe scripts\verify_onnx_raw_pipeline.py `
  --model artifacts\student_structaware15_qat_512.onnx `
  --raw-npy input\fused_packed_raw.npy --raw-layout packed `
  --sensor-id sidd_GP --mode mfnr `
  --black-level 0 --white-level 1 `
  --fusion-confidence-npy input\fusion_confidence.npy `
  --motion-ghost-npy input\motion_ghost.npy `
  --output output\enhanced_mfnr_mosaic.npy --output-layout mosaic `
  --report evidence\mfnr_onnx_report.json
```

输出是归一化 Bayer RAW，不是 RGB 图。后续 ISP 应按同一 CFA、黑白电平、白平衡和镜头校正
参数做 Demosaic/传统处理；不要把输出直接按 sRGB 显示后再反馈为 RAW 评价。

## 8. 目标麒麟 9000 交付与最终 Gate

拿到目标产品团队提供的 HiAI/CANN DDK、转换器、设备系统/固件版本后：

1. 复制 `deploy\hiai\converter.args.example` 到 `evidence\`，按 DDK 文档填写参数；
2. 使用 `deploy\hiai\invoke_converter.ps1` 生成 OM，并保存完整转换日志；
3. 在目标机型测 384/512/640 Tile 的冷态/热态 P50/P90/P99、峰值内存、功耗与稳定性；
4. 补齐目标 Sensor、模式、ISO 的画质结果、失败样本、OM SHA256 与兼容矩阵；
5. 填写 `resources\release_evidence.example.yaml` 的副本并运行 `check-release`。

```powershell
.\.venv\Scripts\isp-ai.exe check-release `
  --evidence evidence\kirin9000_release.yaml `
  --output evidence\kirin9000_release.report.json
```

只有总体状态为 `PASS` 才能称为麒麟 9000 部署通过；缺失 DDK/设备/证据时 `BLOCKED` 是正确结果。

## 9. 每个阶段必须归档的文件

- Git commit、CUDA 环境、精确 YAML、实际命令和开始/结束时间；
- 数据导入回执、Teacher/Student/剪枝/QAT checkpoint SHA256；
- 每阶段 validation JSON 与 `history.jsonl`；
- ONNX、ONNX manifest、ONNX 审计和 DDK 转换日志；
- 真机 profiler、热态曲线、功耗、内存、失败样本及 Release Gate 报告。

将新问题按“现象 → 根因 → 处理 → 防复发”追加到 `docs/ENGINEERING_LOG.md`，并把完成状态更新到
`docs/PROJECT_STATUS.md` 与 `docs/VERIFICATION_REPORT.md`。
