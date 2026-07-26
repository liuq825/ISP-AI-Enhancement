# 训练与恢复规范

## 启动顺序

训练器在创建输出权重前依次执行：

1. 校验 Manifest 文件、重复 ID 和场景级划分泄漏；
2. 按声明的 `data_policy.purpose` 检查数据目录与许可；
3. 确认每个 `sensor_id` 都存在版本化相机嵌入；
4. 构建配对数据、模型、可选教师和特征蒸馏适配器；
5. 恢复 checkpoint 或从固定随机种子开始训练。

商用品质配置还必须声明 `data_requirements`。训练器在创建输出目录前检查 train/val
最少样本、独立物理场景组，以及训练集 Sensor、ISO 桶和模式覆盖。许可合规和文件完整
只说明“可以使用”，不能证明六对样本足以收敛；`configs/train_student_public_baseline.yaml`
因此会主动拒绝当前选择性子集，直到公开训练集扩充到声明下限。

## 混合精度

`amp: true` 只允许 CUDA float16 训练，使用 `torch.autocast` 与
`torch.amp.GradScaler`。梯度裁剪前先执行 `scaler.unscale_`，否则阈值 1.0 会作用于
放大后的梯度。CPU 冒烟配置保持 `amp: false`；不能把 CPU bfloat16 的结果直接当作目标
GPU/端侧 FP16 结论。

## 学习率

省略 `scheduler` 时保持常数学习率。商用品质训练配置使用 epoch 级余弦退火：

```yaml
scheduler:
  type: cosine
  t_max: 100
  eta_min: 0.000001
```

调度器在一轮内全部 `optimizer.step()` 完成后调用一次。训练日志同时记录当前学习率。

## Checkpoint 与恢复

格式版本 2 保存：

- 模型、优化器、调度器、AMP scaler 和可选蒸馏适配器状态；
- 已完成 epoch、global step 与最佳验证 PSNR；
- Python、NumPy、PyTorch CPU/CUDA 和 DataLoader Generator 随机状态；
- 完整训练配置与创建时间。

文件先写同目录临时文件，再原子替换目标。`best.pt` 保存最佳验证模型，
`epoch_NNNN.pt` 按 `save_every_epochs` 保存，并始终保留最终轮。

恢复示例：

```yaml
epochs: 100
resume_checkpoint: runs/student_distill/epoch_0040.pt
```

`epochs` 表示最终总轮数而不是再训练轮数。调度器或蒸馏结构与 checkpoint 不一致时，
训练器拒绝恢复。单元测试从第一轮 checkpoint 恢复第二轮，并要求权重与连续训练逐元素
完全一致。

训练集裁剪随增强随机变化，验证集裁剪固定为中心窗口，因此同一权重不会因验证裁剪
坐标变化而产生分数漂移。多进程 DataLoader 默认不启用 `persistent_workers`：
常驻 worker 内部的 Python/NumPy/Torch 随机状态无法随 checkpoint 保存；每轮重建
worker 后，已保存的 DataLoader Generator 才能精确派生下一轮增强种子。该选择会增加
少量 epoch 启动开销，但保证中断恢复的实验可比性。

## QAT 微调

QAT 必须从已经完成物理剪枝和 FP32 微调的权重开始：

```yaml
model_config: configs/model_student_pruned15.yaml
initial_checkpoint: checkpoints/student_pruned15.pt
qat_config: configs/qat.yaml
```

`initial_checkpoint` 只载入模型权重并开始一个新优化过程；`resume_checkpoint` 恢复同一
训练过程的全部状态，两者互斥。观察器在 `observer_warmup_steps` 后冻结；模型进入
`eval()` 时也不会使用验证数据更新尺度。训练日志记录转换/排除卷积、按层覆盖率和按权重
元素覆盖率。

这些覆盖率仍不是目标 NPU 的 INT8 计算占比。最终 95% 目标只能由准确 HiAI CANN DDK
转换后的节点落点与 profiler 按计算量核验，不能用 Python 卷积层数代替。

QAT checkpoint 的 per-output-channel 权重观察器在建图时即按卷积输出通道定形，
因此断点恢复和导出可以使用 `strict=True`；不能等首批 forward 后才把 scale 从长度 1
扩成输出通道数。导出时必须同时提供 `--qat-config configs/qat.yaml`，以训练时相同
排除规则重建 QAT 图。当前 ONNX Q/DQ 路径只接受激活/权重均为 8 bit；其他训练模拟
位宽会明确拒绝导出。

PyTorch 官方参考：[AMP](https://docs.pytorch.org/docs/stable/amp)、
[可复现性](https://docs.pytorch.org/docs/stable/notes/randomness.html)、
[CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)。
