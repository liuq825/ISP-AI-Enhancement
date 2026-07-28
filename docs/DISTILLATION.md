# Feature + Attention 联合蒸馏方案

## 目标

旧方案主要让 Student 拟合 Teacher 的最终四通道输出，监督信号集中在图尾，难以直接
约束编码器对中尺度纹理、边缘和空间噪声分布的表征。当前方案保留低权重输出约束作为
辅助，同时把主要蒸馏能力升级为多尺度 feature + 空间 attention。

所有蒸馏适配器只在训练时存在；导出 ONNX/OM 时只保留 Student，不增加麒麟 9000
推理参数、算子或时延。

## 三路损失

总损失为：

```text
L = L_gt
  + 0.10 * L_teacher_output
  + 0.15 * L_teacher_feature
  + 0.10 * L_teacher_attention
```

其中 `L_gt` 仍由 Charbonnier、梯度和 RAW 色比真值监督组成。配置权重是公开数据起点，
不是未经消融即可冻结的商用结论。

### 输出辅助蒸馏

Student 和 Teacher 的 RAW residual 分别加回 noisy RAW 并裁剪到 `[0,1]`，再以
Charbonnier 约束增强结果。权重从旧方案的 0.30 降为 0.10，防止 Student 过度复制
Teacher 的系统偏差。

### Feature 蒸馏

层位：

```yaml
feature_keys: [enc2, enc3, enc4, middle, dec2]
```

每个层位使用训练期 1×1 卷积把 Student 通道投影到 Teacher 通道，然后计算
Smooth-L1。Teacher 特征始终停止梯度。Student 的 enc3 从 4 个 block 增加到 6 个
不会改变 stage 输出的空间尺寸和通道接口，因此仍能与 Teacher 的 stage 边界特征对齐。

### Attention 蒸馏

层位：

```yaml
attention_keys: [enc3, enc4, middle, dec1, dec2]
```

对特征 `F ∈ R^(C×H×W)` 计算通道均方空间能量：

```text
A(F) = mean_c(F²)
Â(F) = A(F) / max(||A(F)||₂, epsilon)
```

Student 与 Teacher 的 `Â(F)` 使用 MSE 对齐。该表示与通道数无关，强调网络正在关注的
纹理、边缘和噪声残差区域；全零特征通过 epsilon 保持有限值。实现参考 Attention
Transfer 的空间注意力思想，但采用均方聚合以适配 RAW 恢复特征。

## 配置与恢复

正式入口为 `configs/distill.yaml`：

```powershell
.\.venv\Scripts\isp-ai.exe train --config configs/train_teacher.yaml
.\.venv\Scripts\isp-ai.exe train --config configs/distill.yaml
```

配置使用已经审计的 `data/sidd_training/manifest.jsonl`，并继承 P0 的样本、源配对、
场景、相机和 ISO 数据门槛。第一条命令直接生成第二条引用的
`runs/teacher/best.pt`。Teacher checkpoint 缺失时训练会在启动前失败，不会把普通
监督训练误报成蒸馏。正式配置用 micro-batch 1 与 8 步梯度累积控制峰值显存。

feature 投影参数与 attention 层位语义标记都进入 `distiller_state`。恢复 checkpoint
时采用 `strict=True`：改变 feature key 会造成投影状态不匹配，改变无参数的 attention
key 也会造成语义 buffer 状态键不匹配，禁止静默改变中断前后的蒸馏配方。

训练日志必须同时出现：

- `loss_teacher_output`
- `loss_teacher_feature`
- `loss_teacher_attention`

缺少任一项都表示实际运行配方与方案不一致。

## 消融与放行

在冻结正式配方前至少比较：

1. 真值监督；
2. 真值 + 输出；
3. 真值 + feature；
4. 真值 + attention；
5. 真值 + 输出辅助 + feature + attention。

除总体 PSNR/SSIM 外，必须检查 Sensor×模式×ISO 桶、平坦区残噪、文字/毛发边缘、
彩色噪点和纹理幻觉。只有联合方案在最差域不退化并改善目标难例，才允许作为 P0
蒸馏基线。

参考：

- [Paying More Attention to Attention（Attention Transfer）](https://arxiv.org/abs/1612.03928)
- [Attention Transfer 官方实现](https://github.com/szagoruyko/attention-transfer)
