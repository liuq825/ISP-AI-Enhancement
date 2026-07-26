# 数据集目录与使用策略

机器可读清单位于 `resources/datasets.yaml`，检查日期为 2026-07-26。

## 已核验来源

- SIDD：官方页面说明数据集与关联代码采用 MIT License，包含约 30,000 张真实噪声
  图像及 RAW-RGB/元数据，可用作管线和研究基线。
- SID：官方仓库采用 MIT，但数据文件权利范围仍需单独确认；默认不自动导入。
- DND：官方许可明确限定非商业用途并禁止再分发，只能隔离做研究评测。
- ELD：公开材料中的数据许可不够明确，默认禁用，直到获得书面许可。

当前交付是“达到商用级技术指标的非商业工程验证模型”，不是直接商业发布，因此
商业版权/IP 清关不作为本项目完成前置条件。数据仍不会提交进 Git，且必须遵守来源
自身的用途、注册和禁止再分发条款。未来若直接商业发布，应重新做 IP 审计，并优先
使用目标 Sensor 自有数据重训。

## SIDD 导入

官方 SIDD RAW-RGB `.MAT` 实际是黑电平已扣除、归一化到 `[0,1]` 的二维 Bayer
mosaic，并不是本项目所需的四通道 canonical RAW。`import-sidd` 根据官方 CFA 表
执行以下映射：

`GP=BGGR, IP=RGGB, S6=GRBG, N6=BGGR, G4=BGGR`

转换结果固定为 `[R, Gr, Gb, B]`，同时记录原始文件 SHA256、ISO、曝光分母、CCT、
CFA 与逐场景 NLF。数据切分使用三位 `scene_id` 分组；同一物理场景即使由不同相机
或 ISO 拍摄，也不会跨越 train/val/test。

CLI 默认读取 `resources/sidd_validation_scenes.yaml` 并执行 held-out 硬隔离。
若解压目录意外包含官方 40 个 benchmark 场景中的任意一个，导入会在创建输出目录前
失败。不能通过重新随机切分把这些场景改名为训练集，否则公开验证结论已经泄漏。

`noise_sigma` 根据官方 NLF 在线性 18% 参考灰处计算
`sqrt(mean(beta1*0.18 + beta2))`。它只是 16 通道上下文的可复现标量摘要，完整
六个 NLF 系数仍保存在 Manifest 元数据中。

### 从超大场景 ZIP 选择性获取训练帧

SIDD Full 的单个场景 noisy/GT ZIP 可分别超过 3 GB 和 8 GB，但当前阶段每个场景
只需一对可审计 RAW 做数据链路与初步训练验证。`fetch-sidd-subset` 使用
`remotezip==0.12.3` 读取 ZIP 中央目录，再通过 HTTP Range 只传输目标成员的压缩
字节，而不把整包下载到本地：

```powershell
.\.venv\Scripts\isp-ai.exe fetch-sidd-subset `
  --config resources/sidd_training_subset.yaml `
  --output datasets/SIDD_Training_Subset
```

版本化配置当前覆盖五款相机，并额外加入 `scene_id=010` 验证锚点。下载器先完整校验
配置及 held-out 名单，再顺序访问公共镜像；每个成员必须精确匹配
`<instance>_{NOISY|GT}_RAW_010.MAT`。ZipExtFile 完整读取时验证 ZIP CRC，落盘后再次
核对大小/CRC 并记录 SHA256，最终文件和场景/集合收据都采用原子替换。网络中断后重跑
会复核并复用完整文件；残缺 `.partial` 不会被当作训练数据。

选择性子集只够验证真实数据训练闭环，不等于覆盖 SIDD Full，更不能替代目标 Sensor
数据。默认 `split_seed=20260726` 下，物理 `scene_id=010` 进入 val；配置中的
`scene_id=001` 会进入 test，避免子集全部落入 train。扩充配置时应先检查物理场景分布，
而不是只按相机或实例编号挑选。导入时复用官方验证包内的
`noise_level_functions.csv`，避免 16 通道输入中的噪声强度上下文退化成未知值 0。

`build-sidd-range-config` 进一步读取官网场景表，以及独立的 Mirror 1
`SIDD_URLs.txt` 和 Mirror 2 `SIDD_URLs_Mirror_2.txt`。它逐 `<tr>` 提取 200 个
场景，排除其中 40 个 held-out，再要求剩余 160 个场景在每份清单中都严格对应
`160×5=800` 个角色 URL。仓库已生成 `resources/sidd_medium_range.yaml`，选择
frame 010/020，共 320 对，规模与官方 SIDD Medium 相同但帧选择由本项目显式固定。
配置保存三份来源 SHA256，任一上游数量或顺序变化都会失败，不能靠“URL 可下载”
猜测身份。

双帧获取时，同一场景的 noisy/GT 归档各只打开一次，再在归档内依次校验两个成员；
相比逐配对打开可少 320 次中央目录请求。长时间后台任务可用
`--progress-file outputs/sidd_medium_download.log` 记录已完成场景。预计 RAW 本体
约 20 GB，仍由 `.gitignore` 隔离。`sidd-fetch-status` 利用已完成 pair 收据快速
报告场景/配对百分比、残留 partial 和结构错误，不为查看进度重复计算数百个大文件哈希。
主 CodaLab 镜像发生连接异常时，同一轮立即尝试 York University Mirror 1；收据记录
实际使用的一对 URL。离线恢复只接受当前版本配置列出的主/备 URL 对，不能把任意第三方
镜像混入已验证数据。

训练导入使用 `--patch-size 256 --patches-per-pair 16`。每个 noisy/GT 源配对只加载
和 CFA 打包一次，再按 `patch_seed` 生成 16 个不重复 packed RAW 坐标，避免训练时
每取一个小 crop 都重新解压约 50 MB 的全分辨率 NPZ。Manifest 同时记录
`source_pair_id`、坐标、patch 大小/seed 和两份源 SHA256。数据门禁分别统计 patch
记录与唯一源配对；重复派生更多 patch 不能伪装成更多独立拍摄。

### SIDD RAW 验证块

官方 `ValidationNoisyBlocksRaw.mat` 与 `ValidationGtBlocksRaw.mat` 的变量形状均为
`40×32×256×256`：40 个 held-out 场景、每场景 32 个 Bayer block。MAT 文件本身
没有携带相机代号或 CFA，直接把全部块当 RGGB 会污染 4/5 的相机数据。

`resources/sidd_validation_scenes.yaml` 按官方场景页面的展示顺序固化 40 个
“Held for benchmark”场景。`import-sidd-blocks` 先核验两个 MAT 的变量名、shape、
有限值和 `[0,1]` 范围，再按每个场景的相机 CFA 转换为 `4×128×128` canonical
packed RAW，并记录两个源 MAT 及场景顺序文件的 SHA256。转换时 noisy 与 GT 分开
加载，避免同时常驻约 670 MB 未压缩数组。

本次实际下载、SHA256、转换计数和 noisy baseline 已固化在
`resources/sidd_validation_receipt.yaml` 与 `docs/SIDD_VALIDATION_BASELINE.md`。
数据本体和转换 NPZ 仅本地保存，不提交 Git。

## 为什么公开集不能替代产品数据

公开集无法覆盖目标设备的 CFA、黑电平、模拟/数字增益链、行列噪声、温度漂移、
镜头阴影、坏点、HDR/MFNR 融合残余以及实际固件版本。只在公开集上获得高 PSNR
不能证明已经达到目标麒麟 9000 设备上的商用级画质与性能指标。

## Manifest 约束

每条 JSONL 记录至少包含：

`sample_id, dataset_id, input_path, target_path, split, sensor_id, mode,
session_id, scene_id, iso_bucket, metadata`

切分以 `session_id + scene_id` 为最小隔离组。相邻帧、同一 burst 或同一标定序列
不得跨 train/val/test。`validate-manifest` 会拒绝重复 ID、缺失文件和组级泄漏。

## 烟雾数据

`make-smoke-data` 生成完全程序化、无外部版权依赖的 packed RAW。它包含
shot/read、行、列和黑电平漂移噪声，只用于测试代码和 CI，不能用于画质结论。
