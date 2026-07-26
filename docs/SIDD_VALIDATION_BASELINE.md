# SIDD RAW 验证基线

## 数据证据

2026-07-27 完成官方 `ValidationNoisyBlocksRaw.mat` 和
`ValidationGtBlocksRaw.mat` 下载。两个文件分别按 HTTP 响应长度、SHA256、MATLAB
变量名、shape 和 dtype 五项校验，机器可读收据见
`resources/sidd_validation_receipt.yaml`。

| 文件 | 字节数 | SHA256 | 变量 |
|---|---:|---|---|
| Noisy | 257,564,539 | `25347a0d…fa448` | `40×32×256×256 single` |
| GT | 308,401,493 | `3ea11bc0…76d49` | `40×32×256×256 single` |

转换结果为 1,280 条 test 记录和 2,560 个 NPZ，每块从 Bayer
`256×256` 按对应相机 CFA 打包成 canonical `[R, Gr, Gb, B]`
`4×128×128 float32`。五款相机各 256 块；Manifest 校验无重复 ID、缺失文件或
split 泄漏。

## 未增强输入基线

统一评测入口在 CPU 上耗时 53.73 秒。以下数值衡量 noisy 输入本身，不是模型成绩：

| 域 | 样本 | PSNR (dB) | packed RAW SSIM |
|---|---:|---:|---:|
| Overall | 1,280 | 37.1865 | 0.730949 |
| G4 | 256 | 39.2646 | 0.876817 |
| GP | 256 | 30.1074 | 0.556421 |
| IP | 256 | 49.6152 | 0.888649 |
| N6 | 256 | 35.4253 | 0.726259 |
| S6 | 256 | 31.5200 | 0.606598 |
| Medium ISO | 512 | 38.9834 | 0.869814 |
| High ISO | 608 | 38.4129 | 0.691658 |
| Extreme ISO | 160 | 26.7763 | 0.435886 |

Sensor 间约 19.5 dB 的跨度证明总体均值不足以放行模型。后续 P0、剪枝和 QAT/OM
必须逐 Sensor×模式比较相同 1,280 块；任一域失败不得由 IP 等高分域抵消。

## 口径边界

PSNR 对 canonical packing 前后的像素排列不敏感；本项目 packed RAW PSNR 可稳定用于
模型对照。SSIM 在四个颜色平面分别使用高斯窗口，与官方直接在 Bayer mosaic 上的
窗口邻域不同，因此字段明确命名为 `packed_raw_ssim`，不冒充官方 SIDD SSIM。

公开 SIDD 仍不覆盖目标产品的 Sensor、黑白电平、LSC、HDR/MFNR 融合残余、温度、
固件和 ISP 接口。这份基线只完成 M1 公开验证域，不代表麒麟 9000 商用级 Gate 已通过。
