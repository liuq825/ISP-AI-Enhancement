# SIDD Medium 规模 RAW 获取与导入报告

日期：2026-07-27

## 结论

基于 SIDD 官方 160 个非 held-out 场景，每个场景固定提取 frame 010/020，已经完成
320 对 noisy/GT RAW 的远程 Range 获取、逐成员 CRC/SHA256 校验和最终全量复核。
数据本体只保存在本地，逐配对内容摘要保存在
`resources/sidd_medium_receipt.yaml` 并进入 Git。

| 项目 | 结果 |
|---|---:|
| 场景数 | 160 |
| noisy/GT 配对数 | 320 |
| MAT 文件数 | 640 |
| MAT 总字节数 | 21,618,970,008 |
| 远端目标成员压缩字节数 | 21,506,777,029 |
| CodaLab 主来源配对 | 8 |
| York University 备用来源配对 | 312 |
| 遗留 `.partial` | 0 |

## 证据哈希

- 获取配置 SHA256：
  `a578361b5d1ef3d04b531fa4d5c0b8ac5ebca00adec6d314221bb1dcbcf0ae43`
- 本地集合收据 SHA256：
  `cd32b05b01c58283f9a9c97a6ed50bd9b97263fc777c6208d90c387e443b0e99`
- 可提交逐配对审计收据 SHA256：
  `cd1b74d6cb54a618d5b7dd792af94e27a2495f25d353e5d7e3b62c20b89c4aa2`
- 官方场景页 SHA256：
  `58ec0b448178dc1707dcef8ae18f16504f78c79bfacbbe4df92784d2d8132c11`
- 官方 Mirror 2 清单 SHA256：
  `1e5d7b020f113c900e2e6bc6cd5d092a5304962b0bb0b6149049e982ae264e7b`
- 官方 Mirror 1 清单 SHA256：
  `839db586a28ca8dd914f2fbbe006142611239cd230df2124c5e6cd7668193dc9`

最终审计重新读取 640 个 MAT，验证每个文件的大小、CRC32 和 SHA256，并核对
320 个 pair 收据、集合收据、配置 SHA 及实际主/备 URL。快速状态命令显示
`completed_scenes=160`、`completed_pairs=320`、`status=complete`。

## patch 导入与审计结果

使用固定参数 `patch_size=256`、`patches_per_pair=16`、`patch_seed=20260727` 完成
导入。每个源配对只加载和 CFA 打包一次，再生成 16 个不重复坐标；实际导入耗时
998.3 秒。

| 项目 | 结果 |
|---|---:|
| Manifest 记录 | 5,120 |
| NPZ 文件 | 10,240 |
| NPZ 压缩字节 | 6,155,310,343 |
| 解压 float32 数组字节 | 10,737,418,240 |
| train/val/test patch | 3,296 / 96 / 1,728 |
| train/val/test 源配对 | 206 / 6 / 108 |
| train/val/test 物理场景组 | 6 / 1 / 3 |
| 训练 Sensor | G4 / GP / IP / N6 / S6 |
| 训练 ISO 桶 | low / medium / high / extreme |

最近一次 `audit-sidd-import` 在 220.5 秒内逐个解压 10,240 个 NPZ，检查唯一 `raw` 字段、
float32、4 通道、有限值、`[0,1]` 范围、noisy/GT shape、patch 连续编号和源文件
SHA256，并执行正式 P0 数据门槛。全部通过：

- Manifest SHA256：
  `e9b2f305f72119b13ddae458f24fe179726a7adb564731617d5548de5f03189a`
- 解压数组内容 SHA256：
  `d9dba6d6ac422d8cc0cf20f9b69f9bd21e019f4c5b5925d543b360d47dae3876`
- 导入审计回执 SHA256：
  `d26597ad6ade7f3330a9714bd563037966277e0776fdf792e16b120bfa6ba183`

回执为 `resources/sidd_medium_import_receipt.yaml`，NPZ 和 Manifest 本体仍只保存在
本地。对真实样本 `sidd_0034_001_p001` 的 Student 前向得到
`1×16×256×256 -> 1×4×256×256`。升级后的 `[2,2,6,8]` Student 含
14,586,340 参数，输出全部有限；0.4500 秒是当前 CPU 单次冒烟，不能当作麒麟 9000
时延或画质结论。

## 实际踩坑

1. CodaLab 在第 5 场景开始持续代理断连，仅增加同源重试不能解决。
2. 官方 Mirror 1 支持 HTTP Range，因此加入主备 URL 与实际来源收据；默认仍保留
   HTTPS 主源优先，故障期间显式使用 `--prefer-fallback`。
3. Mirror 1 偶发 502 或 120 秒读取超时；双源轮换、12 次场景重试和线性退避能够
   自动恢复，没有人工删除或覆盖完整文件。
4. 官方 0032 noisy/GT ZIP 将实例号放在父目录而非 basename。兼容逻辑只接受父目录
   精确包含 `0032_NOISY_RAW` / `0032_GT_RAW`，防止通用文件名掩盖错场景归档。
5. 每次重启都先对完整场景重算本地 SHA256/CRC，确认后完全离线复用；未完成成员使用
   `.partial` 写入并在成功后原子替换。
6. 当前 PowerShell 环境同时暴露 `Path`/`PATH`，`Start-Process` 在创建业务进程前
   因字典重复键失败；确认无业务产物后改用前台 Python 命令完成导入。

更完整的设计取舍和防复发措施见 `docs/ENGINEERING_LOG.md`。

## 可重复命令

导入并复核 5,120 条 packed RAW patch 记录：

```powershell
.\.venv\Scripts\isp-ai.exe import-sidd `
  --source datasets/SIDD_Medium_Range `
  --output data/sidd_training `
  --nlf-csv datasets/SIDD_Blocks/noise_level_functions.csv `
  --patch-size 256 --patches-per-pair 16 --patch-seed 20260727
.\.venv\Scripts\isp-ai.exe audit-sidd-import `
  --manifest data/sidd_training/manifest.jsonl `
  --training-config configs/train_student_public_baseline.yaml `
  --acquisition-receipt resources/sidd_medium_receipt.yaml `
  --nlf-csv datasets/SIDD_Blocks/noise_level_functions.csv `
  --output resources/sidd_medium_import_receipt.yaml
```

稳定物理场景切分实际为 train/val/test 源配对 `206/6/108`，对应 patch 记录
`3,296/96/1,728`，与预期一致。

## 边界

这份证据证明公开 SIDD 数据获取和训练输入完整，不证明公开域已经训练收敛，也不能
替代目标 Sensor、固件、融合模式、温度覆盖及麒麟 9000 实机性能证据。
