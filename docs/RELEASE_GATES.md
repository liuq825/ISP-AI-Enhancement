# 量产发布 Gate

任何平均值都不得抵消单个 Sensor/模式失败。

| Gate | 硬性要求 |
|---|---|
| 数据 | 许可、来源、切分、GT、元数据和 Golden Set 均冻结 |
| P0 | 每个 Sensor×模式独立通过 FP32 画质基线 |
| 剪枝 | 物理图参数和索引检查通过；P3 对 P0 域级 PSNR 下降不超过 0.10 dB |
| QAT | 目标 OM 对剪枝 FP32 域级 PSNR 下降不超过 0.08 dB |
| 最终画质 | 对 P0 综合下降不超过 0.15 dB，SSIM、尾部和盲评无显著恶化 |
| 算子 | 按计算量计 INT8 主干目标不低于 95%；回退与 Cast 均有解释 |
| 性能 | 目标分辨率热态 P90 不高于 150 ms，P99 不高于产品门限 |
| 内存 | 融合缓存释放后测 AI 峰值，满足产品预算 |
| 稳定 | 长稳、20 张连拍、切摄、前后台、异常输入无崩溃/超时 |
| 回退 | 故障注入下 100% 出图并记录 reason code |
| 发布 | Model Card、SHA256、manifest、校准包和兼容矩阵签字锁版 |

时延和内存数字是工程起始门限，产品 SLA 更严格时以产品 SLA 为准。

Python QAT 报告中的“按层覆盖率”或“按权重元素覆盖率”不等于本表的按计算量占比。
算子 Gate 必须使用目标 DDK 转换结果和 profiler 统计。

## 机器执行入口

`resources/release_evidence.example.yaml` 是未验证模板，必须替换为目标机真实证据；
模板中的 `false`、空路径和零值不能用于放行。执行：

```powershell
.\.venv\Scripts\isp-ai.exe check-release `
  --evidence evidence/kirin9000_release.yaml `
  --output evidence/kirin9000_release.report.json
```

结果使用三态语义：

- `PASS`：全部 Gate 有证据且逐项通过；
- `FAIL`：已经测量，但至少一个硬阈值或哈希检查失败；
- `BLOCKED`：目标设备、DDK、文件、签字或其他必填证据缺失。

命令只有在总体 `PASS` 时返回退出码 0。报告绑定输入 YAML 的 SHA256，并核验 Model
Card、模型 manifest、OM、校准包和兼容矩阵的真实文件哈希。画质检查逐
`Sensor×模式` 运行，任一域失败都会进入报告，不能由高分域平均抵消。
