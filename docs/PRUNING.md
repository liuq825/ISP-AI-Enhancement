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

## 使用

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[pruning]"
.\.venv\Scripts\isp-ai.exe pruning-summary `
  --source configs/model_student.yaml `
  --target configs/model_student_pruned15.yaml `
  --backend torch-pruning
```

当前参考规格的图统计结果：

| 项目 | 数值 |
|---|---:|
| 源参数量 | 14,348,516 |
| 目标参数量 | 12,176,868 |
| 物理参数剪枝率 | 15.135% |
| DepGraph 剪枝组数 | 40 |
| DepGraph 组内联动操作数 | 260 |
| 删除的逻辑门控通道 | 1,536 |

这些数值只验证结构。最终 P1/P2/P3 必须从已训练 P0 权重出发，以真实验证集做逐域敏感度
分析和剪枝后微调，不能把随机初始化权重的重要性排序作为商用品质结论。

## 已知坑

1. DepGraph 构建需要全局 Autograd 开启，但示例输入本身无需 `requires_grad=True`。当前
   PyTorch 组合下强制该标志会触发 Torch-Pruning 的 unbind 映射异常。
2. Torch-Pruning 1.6.1 发行包中的模块 `__version__` 仍可能报告 `1.6.0`。审计报告使用
   `importlib.metadata.version("torch-pruning")`，以安装发行包元数据为准。
3. DepGraph 只修改张量相关模块，不会自动更新 SimpleGate 保存的 Python 整数属性。
4. 剪枝率必须以物理图参数量或目标 NPU profiler 为依据，不能使用 mask 中零元素比例。

参考：[Torch-Pruning 官方仓库](https://github.com/VainF/Torch-Pruning)、
[PyPI 发行页](https://pypi.org/project/torch-pruning/)。
