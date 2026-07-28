# 本地开发机资源画像与任务边界

检查日期：2026-07-28

## 当前硬件与软件

| 项目 | 实测 |
|---|---|
| 操作系统 | Windows 11，内核 `10.0.26200` |
| CPU | 11th Gen Intel Core i5-11300H @ 3.10 GHz |
| 逻辑线程 | 8 |
| 物理内存 | 15.79 GiB |
| 检查时可用内存 | 约 6.80 GiB；该值会随前台程序变化 |
| PyTorch | `2.13.0+cpu` |
| CUDA | 不可用；`torch.cuda.device_count()=0` |
| NVIDIA 工具 | `nvidia-smi` 不存在 |
| C 盘 | 118.93 GiB，总剩余 18.55 GiB |
| D 盘 | 337.23 GiB，总剩余 292.10 GiB |

Windows CIM 查询在当前权限下返回拒绝访问，因此 CPU 名称改从只读注册表获取，内存
使用 .NET `ComputerInfo`，磁盘使用 `System.IO.DriveInfo`。这也是环境采集脚本不能
只依赖 WMI/CIM 的实际踩坑。

## 本机适合执行

- SIDD 下载、CRC/SHA256、MAT→NPZ 转换和 Manifest/数据门槛审计；
- 71 项 CPU 单元/集成测试、Ruff lint 和文档检查；
- `configs/model_smoke.yaml` 极小模型的单轮训练、恢复、蒸馏和 QAT 链路验证；
- 随机初始化全尺寸 Student 的单样本前向、参数统计和 Torch-Pruning 结构检查；
- 小模型静态 ONNX 导出及 ONNX Runtime 数值对照。

## 本机不适合冒充完成

- Teacher 或 `[2,2,6,8]` Student 的 100 epoch 正式训练；
- CUDA FP16 AMP、正式 QAT 收敛和大批量敏感度搜索；
- HiAI CANN 移动 DDK 的 OM 转换；
- 麒麟 9000 NPU 算子落点、热态时延、内存、功耗和稳定性验收。

这些任务必须迁移到具备 CUDA 的训练机或目标麒麟设备。CPU 链路跑通只能证明代码和
数据契约有效，不能用耗时外推 GPU/NPU 性能，也不能把随机初始化输出当作画质结论。

## 本地执行参数

本机 Windows 多进程 worker 会复制 Python 解释器和数据集状态；在仅约 6.8 GiB
即时可用内存下，多个 NPZ 解压 worker 容易造成换页。因此本地烟雾配置固定：

```yaml
device: cpu
batch_size: 1
num_workers: 0
amp: false
model_config: configs/model_smoke.yaml
```

真实六对 RAW 闭环使用 `configs/train_sidd_subset_smoke.yaml`。正式
`configs/train_student_public_baseline.yaml` 和 `configs/distill.yaml` 明确要求
CUDA，不应为了在本机启动而擅自改成 CPU；那只会产生耗时很长但没有正式训练意义的
运行目录。

## 存储安排

数据和运行产物继续位于 D 盘工作区：

- 320 对 SIDD RAW MAT：约 21.62 GB；
- 10,240 个训练 NPZ：约 6.16 GB 压缩体积；
- `.venv`、`runs/`、`artifacts/`、`datasets/` 和 `data/`：全部位于 D 盘。

C 盘只剩 18.55 GiB，不允许把完整数据集、checkpoint 缓存或模型转换临时目录迁入
C 盘。Git 只保存代码、Markdown 文档和小型 YAML/JSON 回执，继续忽略数据本体。

## 开发决策

1. 本地先完成确定性 CPU 测试、真实 patch 前向和结构审计；
2. 正式 CUDA 训练配置保持独立，防止本地 smoke 参数污染 P0；
3. 所有长任务先估算 D 盘空间和峰值内存，提供断点恢复与原子产物；
4. 获得训练机后先记录 GPU、驱动、CUDA、显存和吞吐，再决定 batch/worker；
5. 获得麒麟 9000 设备后单独记录 DDK、固件、温度、频率和 profiler 证据。
