# 项目状态

更新时间：2026-07-28

## 当前结论

仓库已完成 M0“可复现实验基线”并进入 M1 数据接入。目标是得到达到商用级技术
指标的非商业工程验证模型；不代表本项目直接完成商业版权/IP 交付，也不意味着已经
在麒麟 9000 上通过端侧 Gate。

| Gate | 状态 | 证据或缺口 |
|---|---|---|
| M0 工程烟雾测试 | 已完成 | 71 项 CPU 测试、可恢复/QAT/剪枝产物、静态 ONNX/Release Gate |
| M1 公开验证域 | 已完成 | 官方 SIDD 两个 MAT 校验、1,280 块导入及真实 noisy baseline |
| M1 训练数据 | 已完成 | 320 对 RAW 全量审计；5,120 条 patch/10,240 NPZ 逐文件审计，正式数据门槛通过 |
| 本地开发机 | 已建档 | i5-11300H、8 线程、15.79 GiB、PyTorch CPU；本地仅承担数据/测试/smoke |
| P0 Student FP32 | 等待 CUDA | 已升级 `[2,2,6,8]` 和 feature+attention 蒸馏；真实 patch 前向通过，尚无正式权重 |
| P1/P2/P3 物理剪枝 | 结构感知链路已实现 | 14.953936%、逐 stage 保留率、双后端和来源哈希通过；仍需 P0 权重/逐域敏感度 |
| QAT | 训练/导出链路已接入 | scale 严格恢复与标准 Q/DQ ONNX 已测；最终规则仍须目标 DDK 生成 |
| ONNX | CI 实际导出通过 | 原子产物、ORT 对照、raw16-v1 清单和算子审计已测；仍需真实 P0 权重 |
| OM | 阻塞于外部环境 | 缺目标 DDK、转换器版本、固件和可访问测试设备 |
| ISP 集成与六项放行 | 阻塞于产品环境 | 缺厂商 ISP 接口、RAW、融合 mask、热态 Profiler |
| 商用级自动放行 | 入口已完成 | 三态 Gate 和证据哈希已测；真实证据未齐前保持 BLOCKED/FAIL |

## 下一关键路径

1. 获得产品 Sensor 清单、RAW dump 权限、黑白电平/CFA/LSC 元数据和融合 mask。
2. 锁定麒麟 9000 目标机型、系统版本、HiAI CANN DDK 与 NPU 算子清单。
3. 在 CUDA 训练机启动公开 RAW Teacher/P0，并按 checkpoint/恢复协议持续记录实验。
4. 获得产品数据后补充目标 Sensor 数据并冻结 Golden Set，建立逐域独立基线。
5. 用收敛权重执行 P1/P2/P3、Torch-Pruning、QAT、OM 和热态实机 Gate。
