# M0 验证报告

初始日期：2026-07-26；最近更新：2026-07-28

## 环境

- Python 3.12.13
- PyTorch 2.13.0 CPU
- NumPy 2.5.1
- ONNX 1.22.0
- ONNX Runtime 1.28.0

## 结果

| 检查 | 结果 |
|---|---|
| Ruff lint | 通过 |
| Pytest | 16 passed |
| 当前 Student P0 参数 | 14,586,340 |
| 当前结构感知 P3 参数 | 12,405,108 |
| 当前物理剪枝率 | 14.953936% |
| 程序化 RAW manifest | 通过文件、ID、split 和泄漏检查 |
| 小模型单 epoch 训练 | 通过；仅为代码烟雾测试 |
| 512×512 静态 ONNX | ONNX Checker 通过 |
| PyTorch/ONNX Runtime 最大绝对误差 | `4.18e-7` |
| ONNX 输入/输出 | 固定 `1×16×512×512 -> 1×4×512×512` |
| 动态 shape 子图 | 无 `Shape/Gather/Mod/Pad/Slice` |

## M1 增量验证

| 检查 | 结果 |
|---|---|
| Pytest | 71 passed |
| 中文说明自动门禁 | 覆盖 `src/` 与 `tests/` 的模块、类和函数 |
| Torch-Pruning 发行包 | 1.6.1 |
| Student 拓扑升级 | `encoder_blocks=[2,2,6,8]`，模型构建仍由 YAML 驱动 |
| DepGraph 结构感知剪枝 | 34 个剪枝组、221 个联动操作，删除 1,504 个逻辑门控通道 |
| 结构感知物理剪枝率 | 14.953936%（14,586,340 → 12,405,108），逐 stage 保留率进入 manifest |
| 剪枝后前向 | `1×16×16×16 -> 1×4×16×16` 通过 |
| Feature + Attention 蒸馏 | 投影特征、归一化空间 attention、Teacher 梯度隔离及语义状态恢复通过 |
| 蒸馏训练闭环 | 输出/feature/attention 三项均进入 history，distiller 状态进入 checkpoint |
| SIDD 验证块导入 | 场景顺序、逐相机 CFA、配对清单与注册表嵌入测试通过 |
| SIDD 官方文件 | 两个 MAT 长度/SHA256/变量/shape/dtype 通过；1,280 条清单验证通过 |
| SIDD noisy baseline | 37.1865 dB / 0.730949 packed RAW SSIM；逐 Sensor/ISO 已记录 |
| SIDD 训练隔离 | 官方 held-out 场景命中即拒绝，且失败前不创建转换目录 |
| SIDD 远程子集 | 精确成员、CRC/SHA256、断点复用、集合收据及联网前配置校验通过 |
| SIDD 真实 RAW 导入 | 6 对、五相机、四个 ISO 桶；train/val/test=3/1/2，Manifest 通过 |
| SIDD 真实训练 smoke | 极小 CPU 模型 1 epoch / 3 step、checkpoint 与验证闭环通过；无画质结论 |
| SIDD Medium 规模配置 | 160 train/40 held-out、800 Mirror 2 URL、10 场景/5 相机、320 对通过 |
| SIDD 双帧 Range | 首尾 frame 020 成员实际核验；每场景只打开 noisy/GT 各一次 |
| SIDD 镜像切换 | 两份官方 800-URL 清单与来源哈希；20/120 秒连接/读取超时、备用源提取及离线收据复用通过 |
| SIDD 成员布局 | 实例号位于 basename 或精确角色父目录的两种官方格式通过；错场景目录拒绝 |
| SIDD 最终审计 | 集合/逐配对收据、实际主备来源及全部 MAT SHA256/CRC32 重算；篡改拒绝通过 |
| SIDD Medium 实际获取 | 160 场景、320 对、640 MAT、21,618,970,008 字节；主源 8 对/备用源 312 对，0 partial |
| SIDD patch 导入 | 实际 5,120 条；train/val/test=3,296/96/1,728，源配对=206/6/108 |
| SIDD NPZ 全量审计 | 10,240 文件、6,155,310,343 压缩字节逐个解压；格式、范围、配对 shape 和内容摘要通过 |
| P0 数据预检 | 五相机、四 ISO 桶、single 模式及样本/源配对/场景下限全部通过 |
| 真实 patch 前向 | `[2,2,6,8]` Student：`16×256×256 -> 4×256×256`，14,586,340 参数，输出有限 |
| 数据裁剪复现 | 训练随机裁剪；验证中心裁剪且不消耗全局 RNG |
| 统一评测 | 逐样本 mask PSNR/packed RAW SSIM、Sensor/ISO 分桶与哈希报告通过 |
| 训练恢复 | 从 epoch 1 恢复的 epoch 2 与连续训练权重逐元素完全一致 |
| 梯度累积 | 完整组和末尾不足组按实际 micro-batch 数归一化，global step 仅计 optimizer 更新 |
| QAT 训练 | FP32 初始权重转换、首批观察器初始化、eval 冻结与量化 buffer 保存通过 |
| QAT checkpoint/ONNX | per-channel scale 严格重建；实际 Q/DQ、Checker 与 ORT 对照通过 |
| 剪枝 checkpoint | 双后端权重一致，目标 YAML 严格重建与来源 manifest 通过 |
| 静态 ONNX CI | 极小模型实际导出、Checker、ORT 对照、算子审计与 v2 manifest 通过 |
| 商用级 Release Gate | PASS/FAIL/BLOCKED、逐域失败与交付文件哈希测试通过 |
| 训练数据充分性 | train/val 样本、物理场景、Sensor、ISO、模式下限测试通过 |

相对误差最大值受接近零的输出影响，绝对误差满足 `atol=1e-4, rtol=1e-3`。

## 未验证

- 以上模型是随机初始化后在 8 个程序化样本上训练一轮的小模型，不具有画质价值。
- 未生成 Teacher、P0/P3 商用 checkpoint。
- 未安装目标 HiAI CANN DDK，未生成 OM。
- 未在麒麟 9000 测算子落点、INT8 比例、热态时延、内存、功耗或回退。
- 未获得目标 Sensor 商用授权数据和 ISP 集成接口。

因此当前证据只放行 M0 工程闭环，不能放行任何商用或设备性能结论。
