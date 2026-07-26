# 模型交付清单

每个 ONNX 必须伴随同名 `.manifest.json`，结构约束见
`resources/model_manifest.schema.json`。清单用于证明“这个文件从哪里来、输入是什么、
通用 ONNX 是否自洽”，不能证明目标麒麟 9000 已支持或达到性能指标。

## 必要证据

- `context_contract_version=raw16-v1` 和 16 个有序通道语义；
- 模型配置、checkpoint、导出配置的路径与 SHA256；
- 静态输入/输出 shape、dtype、四通道 residual 语义和真实参数量；
- ONNX opset、文件 SHA256、PyTorch/ONNX/ORT 版本；
- Checker 结果、PyTorch/ORT 最大绝对/相对误差；
- 完整算子计数、高风险算子和动态 shape 子图审计；
- `UNVERIFIED_UNTIL_TARGET_HIAI_CANN_PROFILING` 部署状态。

导出先写临时 ONNX，只有 Checker、ORT 数值对照和结构审计全部通过才原子替换正式文件；
manifest 同样采用临时文件替换。失败任务不得留下一个看似可交付的半截产物。

## 导出器选择

当前显式使用 `dynamo=False` 的 TorchScript ONNX 导出器，并在清单中记录
`legacy_torchscript_dynamo_false`。PyTorch 已提示该路径未来弃用，但迁移到 dynamo
导出器会改变算子分解和目标 DDK 兼容性；必须在准确 HiAI CANN DDK 上完成 A/B 转换与
数值回归后再切换，不能只为消除警告改变交付图。
