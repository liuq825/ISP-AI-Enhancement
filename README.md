# ISP-AI-Enhancement

面向麒麟 9000 级移动 NPU 的 RAW 域降噪增强工程。项目以 NAFNet 为共享骨干，
覆盖单帧 RAW 与 HDR/MFNR 融合后 RAW，提供数据契约、Teacher/Student、蒸馏、
结构化物理剪枝、QAT 准备、ONNX 导出、重叠 Tile 推理和端侧放行工具。

当前状态是“可复现实验与端侧适配基线”，不是已经通过量产认证的模型。真正的商用
发布仍必须使用具备商业授权的目标 Sensor 数据，在目标麒麟 9000 设备、固件和
HiAI CANN DDK 上完成算子、画质、热态时延、内存、稳定性与回退六项验证。

## 快速开始

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\isp-ai.exe model-summary --config configs/model_student.yaml
.\.venv\Scripts\isp-ai.exe make-smoke-data --output data/smoke --samples 16
.\.venv\Scripts\isp-ai.exe validate-manifest --manifest data/smoke/manifest.jsonl
.\.venv\Scripts\python.exe -m pytest
```

训练与导出入口：

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs/train_smoke.yaml
.\.venv\Scripts\isp-ai.exe export-onnx `
  --config configs/model_student.yaml `
  --checkpoint checkpoints/student.pt `
  --output artifacts/student_512.onnx
```

## 输入输出契约

- 输入固定为 `N×16×H×W`，`H/W` 为 packed RAW 尺寸。
- 通道 `0..3` 是 canonical RGGB packed RAW。
- 通道 `4..15` 是噪声、曝光、融合置信度、运动、相机嵌入、白平衡、模式和
  valid mask。
- 网络输出 `N×4×H×W` RAW residual；宿主侧以 FP16/FP32 执行
  `clip(raw + residual, 0, 1)`。

精确定义见 `docs/INPUT_CONTRACT.md`；量产边界与端侧路线见
`docs/DEPLOYMENT_KIRIN9000.md`。

## 重要合规边界

公开数据集的“可下载”不等于“可用于商业模型”。仓库只保存数据源目录、许可快照和
下载/转换入口，不提交数据本体。任何生产权重都必须由法务确认训练数据、基础权重和
工具链许可。详见 `docs/DATASETS.md` 与 `docs/LICENSE_COMPLIANCE.md`。

## 仓库结构

```text
configs/                    可版本化配置
deploy/                     HiAI CANN/OM 与 MindSpore Lite 适配脚本、接口
docs/                       架构、数据、风险、实验和踩坑记录
src/isp_ai_enhancement/     训练与部署核心代码
tests/                      CPU 烟雾测试和契约回归
```
