# ISP-AI-Enhancement

面向麒麟 9000 级移动 NPU 的 RAW 域降噪增强工程。项目以 NAFNet 为共享骨干，
覆盖单帧 RAW 与 HDR/MFNR 融合后 RAW，提供数据契约、Teacher/Student、蒸馏、
结构化物理剪枝、QAT 准备、ONNX 导出、重叠 Tile 推理和端侧放行工具。

当前目标是研发并部署一份达到商用级技术指标的工程验证模型，不是把本仓库及权重
直接作为商业产品交付。项目仍会在目标麒麟 9000 设备、固件和 HiAI CANN DDK 上
完成算子、画质、热态时延、内存、稳定性与回退六项验证；商业发布所需的版权/IP
清关不属于本阶段完成条件。

## 快速开始

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\isp-ai.exe model-summary --config configs/model_student.yaml
.\.venv\Scripts\isp-ai.exe pruning-summary --source configs/model_student.yaml `
  --target configs/model_student_pruned15.yaml --backend torch-pruning
# P0 权重收敛后生成可重建的物理剪枝 checkpoint：
.\.venv\Scripts\isp-ai.exe prune-checkpoint `
  --source-config configs/model_student.yaml `
  --source-checkpoint checkpoints/student_p0.pt `
  --target-config configs/model_student_pruned15.yaml `
  --output checkpoints/student_pruned15.pt
.\.venv\Scripts\isp-ai.exe make-smoke-data --output data/smoke --samples 16
.\.venv\Scripts\isp-ai.exe validate-manifest --manifest data/smoke/manifest.jsonl
.\.venv\Scripts\python.exe -m pytest
```

导入已解压的 SIDD RAW 数据（数据本体不进入 Git）：

```powershell
.\.venv\Scripts\isp-ai.exe import-sidd `
  --source datasets/SIDD `
  --output data/sidd `
  --nlf-csv datasets/SIDD/noise_level_functions.csv
.\.venv\Scripts\isp-ai.exe validate-manifest --manifest data/sidd/manifest.jsonl
```

导入官方 SIDD RAW 验证块（40 场景 × 32 块，转换后每块为 `4×128×128`）：

```powershell
.\.venv\Scripts\isp-ai.exe import-sidd-blocks `
  --noisy datasets/SIDD_Blocks/ValidationNoisyBlocksRaw.mat `
  --ground-truth datasets/SIDD_Blocks/ValidationGtBlocksRaw.mat `
  --output data/sidd_validation `
  --nlf-csv datasets/SIDD_Blocks/noise_level_functions.csv
.\.venv\Scripts\isp-ai.exe validate-manifest `
  --manifest data/sidd_validation/manifest.jsonl
.\.venv\Scripts\isp-ai.exe evaluate `
  --manifest data/sidd_validation/manifest.jsonl `
  --split test --purpose commercial_grade `
  --output outputs/sidd_validation_noisy_baseline.json
```

训练与导出入口：

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs/train_smoke.yaml
# 有真实剪枝权重和 CUDA 环境后启动 QAT：
.\.venv\Scripts\isp-ai.exe train --config configs/train_qat.yaml
.\.venv\Scripts\isp-ai.exe export-onnx `
  --config configs/model_student.yaml `
  --checkpoint checkpoints/student.pt `
  --output artifacts/student_512.onnx
# 真机证据齐全后执行三态商用级放行 Gate：
.\.venv\Scripts\isp-ai.exe check-release `
  --evidence evidence/kirin9000_release.yaml `
  --output evidence/kirin9000_release.report.json
```

## 输入输出契约

- 输入固定为 `N×16×H×W`，`H/W` 为 packed RAW 尺寸。
- 通道 `0..3` 是 canonical RGGB packed RAW。
- 通道 `4..15` 是噪声、曝光、融合置信度、运动、相机嵌入、白平衡、模式和
  valid mask。
- 网络输出 `N×4×H×W` RAW residual；宿主侧以 FP16/FP32 执行
  `clip(raw + residual, 0, 1)`。

精确定义见 `docs/INPUT_CONTRACT.md`；量产边界与端侧路线见
`docs/DEPLOYMENT_KIRIN9000.md`；统一 PSNR 与分桶口径见 `docs/EVALUATION.md`。

## 代码注释与剪枝

所有 Python 模块、类和函数都必须提供中文说明，C++ 公共接口与部署脚本同样记录所有权、
失败语义和工具链约束；CI 会自动检查 Python 覆盖面。规范见
`docs/CODE_COMMENT_STANDARD.md`。

结构化剪枝默认使用 `torch-pruning==1.6.1` 的 DepGraph 物理删除通道，并保留手工重建
后端作为交叉验证。SimpleGate 成对索引、已知兼容性问题和参考结果见 `docs/PRUNING.md`。
AMP、余弦学习率、原子 checkpoint 和确定性恢复约束见 `docs/TRAINING.md`。
ONNX 伴生证据和 JSON Schema 见 `docs/MODEL_MANIFEST.md`。

## 数据使用边界

当前模型用于非商业研发和麒麟 9000 部署验证，可以使用许可允许研究用途的公开
数据集。仓库保存数据源目录和转换入口，不提交数据本体，并继续遵守“不再分发”等
原始条款。未来若把模型直接用于商业发布，需另行完成版权/IP 清关，必要时使用目标
Sensor 自有数据重训。详见 `docs/DATASETS.md`。

## 仓库结构

```text
configs/                    可版本化配置
deploy/                     HiAI CANN/OM 与 MindSpore Lite 适配脚本、接口
docs/                       架构、数据、风险、实验和踩坑记录
src/isp_ai_enhancement/     训练与部署核心代码
tests/                      CPU 烟雾测试和契约回归
```
