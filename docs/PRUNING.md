# 结构化剪枝

## 为什么使用结构化剪枝

部署目标需要真实缩小卷积权重和计算图。`torch.nn.utils.prune` 的默认用法主要为权重添加
mask，即使稀疏权重变成零，通道 shape、参数存储和常规 NPU 卷积量通常不会减少。本项目默认
使用 `torch-pruning==1.6.1` 的 DepGraph 执行物理通道删除，并保留手工重建实现作为交叉
验证基线。

## NAFNet 的特殊约束

NAFBlock 的 SimpleGate 把 `2H` 个物理通道均分后逐元素相乘。删除一个逻辑通道 `i` 时，
必须同时删除左右两半的 `[i, i + H]`。第一分支还联动 depthwise 卷积、SCA 和输出投影，
第二分支联动 FFN 输出投影。只剪一个卷积会产生 shape 错误，或更危险的门控语义错位。

`torch_pruning_adapter.py` 对每个 NAFBlock 构建 DepGraph、提交成对索引并校验联动后的
`conv1/conv2/sca_conv/conv3/conv4/conv5` 通道。DepGraph 修改模块后，还需手动同步
`dw_hidden_channels`、`ffn_hidden_channels` 和两个 SimpleGate 的静态分割长度；这些普通
Python 属性不在依赖图的修改范围内。

## 结构感知 15% 分配

旧方案按近似统一比例缩减各深层 block，无法表达浅层 RAW 细节、stage 边界和 skip
接口的敏感性。当前 `configs/model_student_structaware15.yaml` 使用以下结构先验：

- enc1/enc2/dec3/dec4 完整保留，优先保护高分辨率细节与输出恢复；
- enc3/enc4 的首尾 block 保留更宽，内部 block 承担更多压缩；
- Middle 和深层 Decoder 提供主要压缩预算；
- 所有逻辑隐藏宽度 16 对齐，方便目标 NPU 通道调度；
- 全局目标仍是约 15%，但不要求每层或每块都是 15%。

| Stage | P0 逻辑宽度 | 结构感知目标 | stage 隐藏通道保留率 |
|---|---|---|---:|
| enc1 | 32, 32 | 32, 32 | 100.000% |
| enc2 | 64, 64 | 64, 64 | 100.000% |
| enc3 | 128×6 | 128,112,112,112,112,128 | 91.667% |
| enc4 | 256×8 | 256,224,208,208,208,208,224,256 | 87.500% |
| middle | 512×4 | 448,400,400,432 | 82.031% |
| dec1 | 256×2 | 224,240 | 90.625% |
| dec2 | 128×2 | 112,128 | 93.750% |
| dec3 | 64×2 | 64,64 | 100.000% |
| dec4 | 32×2 | 32,32 | 100.000% |

上述是进入真实敏感度实验的参考结构，不是假定的最终最优解。正式 P1/P2/P3 应对每个
候选 block 做“暂时裁剪→短微调→逐域验证”，在任一 Sensor×模式×ISO 桶超过 PSNR/
SSIM/边缘预算时回退该 block。权重重要性只负责在已分配宽度内选择具体通道。

## 使用

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[pruning]"
.\.venv\Scripts\isp-ai.exe pruning-summary `
  --source configs/model_student.yaml `
  --target configs/model_student_structaware15.yaml `
  --backend torch-pruning
```

命令同时输出全局物理剪枝率和 `stage_hidden_retention`，防止只看 15% 总数掩盖错误的
逐层分配。当前参考规格的真实图统计结果：

| 项目 | 数值 |
|---|---:|
| 源参数量 | 14,586,340 |
| 目标参数量 | 12,405,108 |
| 物理参数剪枝率 | 14.953936% |
| DepGraph 剪枝组数 | 34 |
| DepGraph 组内联动操作数 | 221 |
| 删除的逻辑门控通道 | 1,504 |

这些数值只验证结构。最终 P1/P2/P3 必须从已训练 P0 权重出发，以真实验证集做逐域敏感度
分析和剪枝后微调，不能把随机初始化权重的重要性排序作为商用品质结论。

## 剪枝已训练权重

```powershell
.\.venv\Scripts\isp-ai.exe prune-checkpoint `
  --source-config configs/model_student.yaml `
  --source-checkpoint runs/student_feature_attention_distill/best.pt `
  --target-config configs/model_student_structaware15.yaml `
  --output checkpoints/student_structaware15.pt `
  --backend torch-pruning
```

产物包含源/目标配置和源 checkpoint 的 SHA256、真实参数量、剪枝率与 DepGraph 统计，
逐 stage 隐藏通道保留率也写入 `.pt.manifest.json`。保存前会从目标 YAML 重新创建
独立模型、严格加载剪枝权重并逐元素比较输出，证明计算图可由配置重建。随后使用
`configs/train_structaware15.yaml` 微调画质，通过 P0 对照后才进入
`configs/train_qat.yaml`。

## 已知坑

1. DepGraph 构建需要全局 Autograd 开启，但示例输入本身无需 `requires_grad=True`。当前
   PyTorch 组合下强制该标志会触发 Torch-Pruning 的 unbind 映射异常。
2. Torch-Pruning 1.6.1 发行包中的模块 `__version__` 仍可能报告 `1.6.0`。审计报告使用
   `importlib.metadata.version("torch-pruning")`，以安装发行包元数据为准。
3. DepGraph 只修改张量相关模块，不会自动更新 SimpleGate 保存的 Python 整数属性。
4. 剪枝率必须以物理图参数量或目标 NPU profiler 为依据，不能使用 mask 中零元素比例。
5. 调用 DepGraph 的外层工作流也不能使用 `torch.no_grad`/`torch.inference_mode` 装饰；
   只能在依赖图完成后对重建模型的数值比较局部关闭梯度。

参考：[Torch-Pruning 官方仓库](https://github.com/VainF/Torch-Pruning)、
[PyPI 发行页](https://pypi.org/project/torch-pruning/)。
