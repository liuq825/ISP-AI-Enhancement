# SIDD RAW 选择性训练子集

## 获取与完整性

2026-07-27 从 SIDD 官网逐场景表的 Codalab Raw-RGB Mirror 2 获取 6 个非 held-out
场景的 frame 010。下载器没有传输约数 GB 的完整场景 ZIP，而是读取中央目录后仅以
HTTP Range 获取目标压缩成员。12 个 MAT 共 398,633,116 字节，每个文件均通过成员名、
解压大小、ZIP CRC 和本地 SHA256 四项校验；精确值见
`resources/sidd_training_subset_receipt.yaml`。

场景覆盖 G4、GP、IP、N6、S6 五款相机，并加入 `scene_id=010` 作为稳定 val 锚点。
批量入口在任何网络请求前对照官方 40 个 held-out 场景，实际命中数为 0。公开验证块
继续只存在于独立 test Manifest，不参与本子集训练。

## 导入结果

官方二维 Bayer MAT 依每款相机 CFA 转为 canonical `[R, Gr, Gb, B]` packed RAW；
官方 NLF CSV 为 16 通道输入提供非零噪声强度条件。导入后有 6 条记录和 12 个 NPZ，
共 381,546,431 字节：

| Split | 配对 | 物理 scene_id | 用途 |
|---|---:|---|---|
| train | 3 | 002、003 | 真实链路 smoke |
| val | 1 | 010 | 独立训练验证 |
| test | 2 | 001 | 子集只读对照 |

Manifest SHA256 为
`9ee3eacb3af5728f99f3df9e6aee208e713bb73de76b74f519bafe171873d591`，
文件、ID 和场景级 split 泄漏检查全部通过。

未增强输入在 test 两张上的基线为 31.6284 dB / 0.619623 packed RAW SSIM；单张
low-ISO val 为 43.0296 dB / 0.966171。样本极少，这些值仅用于发现转换回归，不能与
1,280 块公开验证基线或论文结果直接比较。

## 真实数据训练烟雾结果

`configs/train_sidd_subset_smoke.yaml` 使用极小 NAFNet、CPU、128 packed RAW crop
训练一轮，共 3 个 optimizer step，成功生成原子 checkpoint；验证 PSNR 为
13.6686 dB。该结果证明 MAT→CFA 打包→NLF/相机上下文→16 通道→损失→验证→可恢复
checkpoint 的真实数据闭环可以执行。

它不是 P0，更不是达到商用品质的权重。正式
`configs/train_student_public_baseline.yaml` 声明至少 10,000 条 train、300 条 val、
6 个训练物理场景、五款相机和四个 ISO 桶；当前子集会被训练前门禁拒绝。最终商用级
放行还必须加入目标 Sensor 数据，并在目标麒麟 9000 设备完成画质、算子、性能、内存、
稳定性和回退证据。
