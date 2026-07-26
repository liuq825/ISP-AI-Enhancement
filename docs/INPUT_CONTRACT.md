# 16 通道输入契约

张量固定为 `N×16×H×W`，dtype 在训练/ONNX 基线为 FP32，端侧候选为 FP16。
`H/W` 是 packed RAW 尺寸，必须是 16 的倍数；Pad 区由通道 15 标记。

| 通道 | 语义 | 精确定义 |
|---|---|---|
| 0-3 | RAW | 黑白电平归一化后的 canonical `[R, Gr, Gb, B]`，范围 `[0,1]` |
| 4 | noise sigma | 目标 Sensor 标定噪声强度，截断到 `[0,1]` |
| 5 | exposure | `log2(ratio)` 在 `[-8,8]` 线性映射至 `[0,1]`；ratio=1 对应 0.5 |
| 6 | fusion confidence | 单帧为 1；融合路径为与 RAW 对齐的 `[0,1]` 图 |
| 7 | motion/ghost | 单帧为 0；融合路径为与 RAW 对齐的 `[0,1]` 图 |
| 8-11 | camera embedding | 四个常量平面，单值范围 `[-1,1]` |
| 12 | R/G WB | `log2(ratio)` 在 `[0.25,4]` 对数范围映射至 `[0,1]` |
| 13 | B/G WB | 同上 |
| 14 | mode | single=0、HDR=0.5、MFNR=1；版本变更需重训 |
| 15 | valid mask | 有效像素 1，Pad 或无效边界 0 |

`ContextBuilder` 对 NaN、范围、CFA、shape 和缺失 embedding 执行失败即停。量产
运行时不得默默使用未知 Sensor 的默认 embedding；应返回明确错误并回退传统 ISP。

## CFA 规范化

输入 Bayer 平面必须先根据 `RGGB/GRBG/GBRG/BGGR` 映射到固定
`[R, Gr, Gb, B]`。这里的两个绿色通道不可随意平均，因为行列读出噪声和像素响应
可能不同。

## 版本化

以下任一项变化都必须提升契约版本并重新回归：通道语义、归一化区间、mode code、
camera embedding、CFA 映射、noise sigma 单位或 Pad 规则。
