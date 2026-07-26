# 项目状态

更新时间：2026-07-26

## 当前结论

仓库已进入 M0“可复现实验基线”。这意味着数据契约、模型、训练、剪枝、QAT 模拟、
ONNX 导出和 Tile 推理能够在通用 PyTorch 环境中验证；不意味着已经生成商用权重，
也不意味着已经在麒麟 9000 上通过量产认证。

| Gate | 状态 | 证据或缺口 |
|---|---|---|
| M0 工程烟雾测试 | 已完成 | 16 项 CPU 测试、合成 RAW、小模型训练、静态 ONNX 一致性 |
| P0 Student FP32 | 未开始 | 缺目标 Sensor 商用授权训练集与 GPU 训练资源 |
| P1/P2/P3 物理剪枝 | 工具已实现 | 需要 P0 权重、分域验证和梯度敏感度缓存 |
| QAT | 模拟链路已实现 | 最终量化规则必须由目标 HiAI CANN DDK 重新生成 |
| ONNX | 导出链路已实现 | 需要实际 checkpoint 与 ONNX Runtime 对照 |
| OM | 阻塞于外部环境 | 缺目标 DDK、转换器版本、固件和可访问测试设备 |
| ISP 集成与六项放行 | 阻塞于产品环境 | 缺厂商 ISP 接口、RAW、融合 mask、热态 Profiler |

## 下一关键路径

1. 获得产品 Sensor 清单、RAW dump 权限、黑白电平/CFA/LSC 元数据和融合 mask。
2. 锁定麒麟 9000 目标机型、系统版本、HiAI CANN DDK 与 NPU 算子清单。
3. 完成目标 Sensor 数据采集与许可签署，冻结 Golden Set。
4. 训练 Teacher/P0，并在每个 Sensor×模式×ISO 桶建立独立基线。
5. 用真实权重执行 P1/P2/P3、QAT、OM 和热态实机 Gate。
