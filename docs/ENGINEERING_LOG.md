# 工程日志、难点与踩坑

## 2026-07-26：方案审阅与 M0 初始化

### 文档渲染器找不到 LibreOffice

- 现象：标准 DOCX 渲染脚本因缺少 `soffice` 失败。
- 处理：隐藏调用本机 Word 导出 PDF，再用 Poppler 生成 25 页 PNG，逐页检查。
- 结论：原文档无截断、重叠、损坏表格、批注或修订；临时渲染文件不入库。

### 中文路径通过 Python 标准输入后乱码

- 现象：PowerShell here-string 里的中文 DOCX 路径被控制台编码替换为 `?`。
- 处理：改为在工作目录中用 `Path.glob("*.docx")` 发现文件，并显式设置
  `PYTHONIOENCODING=utf-8`。
- 防复发：脚本优先接收 pathlib 路径；Windows CI 不把非 ASCII 路径硬编码进
  stdin 程序。

### 手算 P0 参数时漏计 Depthwise 权重

- 现象：首次手算得到约 14.245M，误以为原方案的 14.347M 不准确。
- 根因：参数公式漏掉 Depthwise 3×3 的 `18C` 权重线性项。
- 处理：代码精确统计为 `14,348,516`，确认原方案近似值正确；15% Gate 以该物理图
  为分母。
- 防复发：评审和 CI 只使用模型图统计，不再用手算常数裁决。

### 原 Middle=400 参考会过剪

- 现象：对所有四个 Middle Block 同时把两个扩展分支降到 400，会超过约 15% 总
  参数剪枝。
- 处理：参考图改为按块 `[416,416,432,432]`；最终仍以分域敏感度重分配。
- 防复发：`pruning-summary` 在 CI 中验证物理参数和目标区间。

### pip 后台安装进程被宿主回收

- 现象：首次 120 秒命令超时；改用 `Start-Process` 后，下载完成但子进程在安装阶段
  被宿主回收，日志没有标准错误。
- 处理：保留 pip 缓存，改回前台命令完成安装。
- 防复发：大依赖先分批下载/安装；不要假设桌面任务结束后后台子进程会持续存活。

### Hann 窗口边界被错误衰减

- 现象：Pointwise 模型的 Tile 输出有 4.5% 像素与整帧结果不一致，误差集中在图像
  外边界。
- 根因：一维 Hann 先截断到 `min_weight`，二维外积后最小值变成
  `min_weight²`，但归一化分母仍截断到 `min_weight`。
- 处理：一维边界改为 `sqrt(min_weight)`，确保二维窗口下界与分母一致。
- 防复发：保留“无感受野 Pointwise 模型的整帧/Tile 严格一致”单测。

### 静态 ONNX 被自动 Pad 逻辑污染

- 现象：首次 512 ONNX 虽然声明静态输入，但输出 shape 是符号值，图中残留大量
  `Shape/Gather/Mod/Pad/Slice`。
- 根因：常规 forward 为任意 H/W 执行运行时 Pad 和裁剪，trace 把它们带进部署图。
- 处理：新增 `forward_static`，只接受宿主已验证的 16 对齐固定 shape；导出器用
  专用 wrapper 调用该路径。
- 防复发：ONNX 审计必须确认输入输出全为整数维度，且不含动态 shape 子图。

### 移动 CANN 与服务器 CANN 不可混为一谈

- 现象：原方案直接写 `ATC/CANN -> OM`，未锁移动 HiAI DDK。
- 处理：OM 仍是量产目标，但转换器和参数由目标产品 DDK 决定；MindSpore Lite
  仅作备用功能验证。
- 放行要求：转换成功不是算子落点、量化比例或时延通过的证据。

### 公开数据集商业许可风险

- 现象：公开 RAW 集许可差异大，DND 明确禁止商业使用。
- 处理：建立机器可读目录并记录 `dataset_id`。当前模型定位为非商业研发与部署
  验证，可在原许可允许的研究范围使用；若未来直接商业发布再执行 IP 清关或重训。

## 2026-07-26：M1 数据接入加固

### 训练器把真实 Sensor 错当成 smoke_sensor

- 现象：训练入口内部硬编码 `smoke_sensor` embedding，配置中的 Sensor 注册表
  没有真正参与训练。
- 风险：真实数据可能在启动后才因未知 Sensor 失败，或被迫把不同 Sensor 伪装成
  同一域。
- 处理：训练配置必须声明 `context_config`；Manifest 中每颗 Sensor 都必须在
  版本化注册表存在，样本内联 embedding 与注册表不一致时失败即停。

### 可下载数据没有用途追溯

- 现象：旧 Manifest 不记录 `dataset_id`，训练器无法区分 SIDD、DND 或自有数据。
- 风险：无法复现实验数据构成，也可能违反数据自身的禁止再分发或用途限制。
- 处理：Manifest 强制记录 `dataset_id`。`commercial_grade` 表示非商业、商用级
  技术验证；严格 `production` 审批门禁仅保留给未来直接商业发布。

### SIDD 的 Raw-RGB 容易被误解成四通道

- 现象：SIDD 页面称其为 Raw-RGB，但文件内容是二维 Bayer mosaic。
- 风险：直接当作 RGB/四通道会导致 CFA 错位；训练仍可能收敛，但颜色和行列噪声
  关系已经被破坏。
- 处理：转换器使用官方五款相机 CFA 表统一打包成 `[R, Gr, Gb, B]`，并按场景号
  进行组级切分，防止内容泄漏。

### `.gitignore` 的 `data/` 误伤 Python 包

- 现象：本地 27 项测试通过，但 Git 索引中没有 `src/isp_ai_enhancement/data/`；
  远端首次提交因此缺失 Context、Manifest 和数据集实现。
- 根因：未锚定的 `data/` 会匹配仓库任意层级同名目录，不只匹配根目录训练数据。
- 处理：改为 `/data/`、`/datasets/` 等根目录锚定规则，并通过 `git ls-files` 验证
  数据源码确实进入索引。
- 防复发：提交前同时执行测试、`git status`、`git check-ignore -v` 和干净检出验证；
  本地通过不能替代对版本控制内容的核验。

## 2026-07-26：中文注释门禁与 Torch-Pruning

### SimpleGate 不能按普通卷积独立剪枝

- 现象：DepGraph 可以跟踪卷积和 SCA，但 SimpleGate 把通道一分为二后逐元素相乘。
- 风险：只删除索引 `i` 会造成左右两半错位；即使 shape 能运行，门控语义也已经损坏。
- 处理：每个逻辑通道使用 `[i, i + hidden]` 成对删除，并校验 depthwise、SCA 和前后
  1×1 卷积的联动 shape。
- 防复发：单测对第一块的所有相关卷积通道和完整前向 shape 做断言。

### DepGraph 不会更新普通 Python 通道属性

- 现象：Torch-Pruning 完成物理删除后，卷积 shape 已正确缩小，但前向仍按旧
  `hidden_channels` 执行 `torch.split`。
- 根因：依赖图只修改张量/模块，不知道 SimpleGate 中整数属性也是图契约的一部分。
- 处理：每次剪枝后同步 `dw_hidden_channels`、`ffn_hidden_channels`、
  `gate1.hidden_channels` 和 `gate2.hidden_channels`。

### Torch-Pruning 的示例输入不应强制梯度

- 现象：在当前 PyTorch 与 Torch-Pruning 1.6.1 组合中，为示例输入设置
  `requires_grad=True` 会在 unbind 索引映射中触发 `UnboundLocalError`。
- 处理：保持全局 Autograd 开启以便 DepGraph 跟踪，但示例输入本身使用默认
  `requires_grad=False`。

### 安装版本与模块版本字符串不一致

- 现象：安装发行包为 1.6.1，但 `torch_pruning.__version__` 返回 1.6.0。
- 处理：模型审计报告改用 `importlib.metadata.version("torch-pruning")`。
- 防复发：测试断言报告版本与发行包元数据一致。

### “详细中文注释”容易在后续提交中退化

- 处理：为 Python 模块、类、函数补齐中文 docstring，为关键 ABI 和成对剪枝逻辑补充
  行内解释；C++ 接口使用中文 Doxygen，PowerShell 使用 comment-based help。
- 防复发：`tests/test_chinese_documentation.py` 扫描 `src/` 与 `tests/`，新增代码缺少
  中文说明时 CI 失败。

### Windows 后台下载把含空格的路径拆成两个参数

- 现象：`Start-Process -ArgumentList` 传递含空格的绝对 `--output` 路径时，curl
  命令行中的引号丢失；目标路径后半段被当成 URL，官方 MAT 内容写入 stdout 日志。
- 处理：停止错误进程，确认 stdout 内容具有 MATLAB 文件头后恢复为 partial 文件；
  重启时设置工作目录并使用不含空格的相对输出路径，再从服务器续传。
- 防复发：后台启动后立即读取实际进程命令行，并同时检查目标文件与 stdout 大小；
  不能只依赖 curl 进度日志判断落盘位置。

### PowerShell 同时存在 Path/PATH 时 Start-Process 拒绝启动

- 现象：SIDD 转换后台任务两次在启动前失败，报环境字典已存在 `Path`/`PATH` 重复键；
  `-UseNewEnvironment` 在当前宿主仍无法规避。
- 处理：使用 `System.Diagnostics.ProcessStartInfo`、`UseShellExecute=true` 和隐藏窗口
  启动同一个虚拟环境解释器；所有参数改为不含空格的工作目录相对路径。
- 防复发：启动后同时监控 venv launcher 与实际 Python 子进程，并以输出文件精确计数
  判断完成，不能只看父 PID。

### 下载完成不等于 MAT 可用

- 处理：分别核验 HTTP 预期长度、SHA256、唯一 MATLAB 变量、`40×32×256×256`
  shape 和 `single` dtype；转换后再验证 1,280 条 Manifest 与 2,560 个 NPZ。
- 结果：noisy 输入基线为 37.1865 dB / 0.730949 packed RAW SSIM；五个 Sensor 分桶
  跨度很大，后续模型放行禁止只看总体均值。

### 验证块 MAT 没有保存相机 CFA

- 现象：两个官方 RAW 验证文件只包含 `40×32×256×256` 浮点数组，没有相机代号。
- 风险：统一按 RGGB 打包时，GP/N6/G4/S6 均会发生颜色通道和行列噪声语义错位。
- 处理：从官方场景页提取按展示顺序标灰的 40 个 held-out 场景，保存为版本化
  `resources/sidd_validation_scenes.yaml`；导入器将其第一维与 MAT 第一维严格绑定。
- 防复发：场景表、noisy 和 GT 数量或 shape 任一不符时失败；输出 Manifest 记录
  三份来源文件的 SHA256。

### 数据集默认零向量再次绕过相机注册表

- 现象：治理预检确认了 `sensor_id` 存在，但 `RawPairDataset` 在清单未写
  `camera_embedding` 时主动构造 `(0,0,0,0)`，导致 `ContextBuilder` 不再查询注册表。
- 风险：所有真实相机在训练时可能共享零嵌入，治理检查通过却没有真正使用配置。
- 处理：清单没有覆盖值时传递 `None`，由 ContextBuilder 按 `sensor_id` 读取注册表。
- 防复发：新增数据集级单测，以非零注册向量逐通道核验最终 16 通道输入。

### 按 batch 平均验证 PSNR 会错误加权

- 现象：旧训练验证先对 batch 内像素求一个 PSNR，再对 batch 数求平均。
- 风险：最后一个较小 batch 与完整 batch 权重相同；不同 batch size 会得到不同指标。
- 处理：每个样本在自身有效 mask 内独立计算 PSNR，再按样本等权汇总。
- 防复发：单测用 20 dB 与 40 dB 两个样本确认最终均值严格为 30 dB；统一评测报告
  同时按 Sensor 与 ISO 桶累加逐样本值。

### 只有 PSNR 不能监控结构退化

- 处理：统一评测增加逐样本 packed RAW 高斯窗 SSIM，并与 PSNR 一起按 Sensor/ISO
  分桶；无效 mask 只有在完整局部窗口有效时才参与。
- 边界：packed RAW SSIM 与官方 Bayer mosaic 实现的窗口采样不同，报告字段明确命名，
  不把内部回归值冒充官方 SIDD 成绩。

## 2026-07-26：训练可靠性加固

### 配置写了 AMP，但训练器完全没有使用

- 现象：`distill.yaml` 声明 `amp: true`，旧循环仍全程 FP32，配置不会报错也没有效果。
- 风险：显存、速度和数值稳定性评估全部失真，日志却让人误以为训练已使用混合精度。
- 处理：CUDA 路径使用 `torch.autocast` 和 `torch.amp.GradScaler`；CPU 收到
  `amp: true` 时直接失败，避免静默忽略。
- 防复发：日志记录 grad scale；梯度裁剪前必须 `unscale_`。

### Checkpoint 只能加载权重，不能可靠恢复训练

- 现象：旧文件虽保存 optimizer，却没有 scheduler、AMP、global step 或随机状态，
  `history.jsonl` 也会在重启时覆盖。
- 处理：格式升级到 v2，保存全部优化与随机状态；恢复时追加日志，并把 `epochs`
  解释为最终总轮数。
- 防复发：集成测试比较“连续第二轮”和“从第一轮恢复后第二轮”的每个权重张量，
  要求 `rtol=0, atol=0`。

### 直接写 checkpoint 可能留下可见的半截文件

- 处理：先写同目录 `.tmp`，写完后原子替换；同时维护 `best.pt` 和按间隔保存的
  epoch 文件。

### 验证随机裁剪与常驻 worker 破坏复现

- 现象：训练和验证共用随机裁剪；相同权重重复验证会抽到不同窗口。多进程
  `persistent_workers` 的内部随机状态也没有写入 checkpoint，恢复后增强序列会分叉。
- 根因：只保存主进程与 DataLoader Generator 状态，不等于保存常驻子进程 RNG。
- 处理：验证/测试固定中心裁剪；训练仍随机裁剪。每轮重建 worker，并由已保存的
  Generator 派生 worker 种子。
- 防复发：新增中心裁剪不消耗 RNG 的单测；确定性恢复配置不启用常驻 worker。

### SIDD 训练目录可能混入官方 held-out 场景

- 现象：仅按哈希切分本地场景，无法阻止误下载的 benchmark 场景被重新分到训练集。
- 根因：数据完整性校验只检查本地跨 split 泄漏，没有对照官方 40 场景名单。
- 处理：`import-sidd` 默认加载版本化 held-out 清单，命中完整场景实例名时在写产物前
  拒绝导入。
- 防复发：单测构造 held-out RAW 对，并验证输出目录不会被创建。

## 2026-07-26：QAT 训练接入

### 激活观察器从固定 1.0 开始会浪费 RAW 精度

- 现象：per-tensor 激活观察器的 `max_abs` 初值为 1.0，首批低幅 RAW 仍只按 EMA
  缓慢向真实范围衰减。
- 风险：QAT 初期 scale 过大，大量有效暗部值落在相同整数格点，影响收敛。
- 处理：增加持久化初始化标志；首个训练批直接复制真实最大绝对值，后续才做 EMA。
- 防复发：单测用幅值 0.125 的首批输入，要求观察范围精确等于 0.125。

### 验证集曾可能修改量化尺度

- 根因：观察器开关没有检查模块的 train/eval 状态。
- 处理：只有 `observer_enabled and training` 时更新范围；验证与推理只复用冻结尺度。
- 防复发：eval 状态输入更大幅值后，测试要求 `max_abs` 保持不变。

### QAT YAML 原先没有接入训练器

- 现象：仓库有 `qat.yaml` 和伪量化模块，但 `train` 从不读取或替换模型。
- 处理：新增 `qat_config` 和 `initial_checkpoint` 路径，从剪枝 FP32 权重构建 QAT
  图，按 global step 冻结观察器，并把量化 buffer 保存进 v2 checkpoint。
- 边界：层数/权重元素覆盖率只用于代码审计，不能替代目标 OM 的按计算量 INT8 落点。

### QAT per-channel scale 无法 strict 恢复

- 现象：权重观察器建图时 `max_abs` 长度为 1，首批 forward 后才变成
  `out_channels`；重建图加载 checkpoint 时出现数十个 shape mismatch。
- 影响：不只 QAT ONNX 导出失败，同一问题也会破坏 QAT 训练断点恢复。
- 处理：`QATConv2d` 创建时即按输出通道定形权重 scale；测试在首批后把状态严格加载
  到一个从未 forward 的新 QAT 模型。

### 自定义 Round/Clip 不等于可部署 INT8 图

- 现象：原伪量化公式若直接 trace，只会产生 `Round/Clip/Mul` 浮点子图，转换器无法
  稳定识别量化意图。
- 处理：8 bit 路径改用 PyTorch 标准 fake-quant 算子；ONNX 实际审计必须出现
  `QuantizeLinear/DequantizeLinear`，且 ORT 与 PyTorch 数值对照通过。
- 边界：Q/DQ 仍只是交付给目标 DDK 的量化意图；INT8 NPU 落点只能由 OM profiler
  证明。

## 2026-07-26：剪枝产物工作流

### 上层 checkpoint 工作流的 inference_mode 会让 DepGraph 失败

- 现象：底层适配器已按要求开启 Autograd，但新增的 `prune-checkpoint` 外层最初使用
  `@torch.inference_mode()`，DepGraph 仍检测到全局梯度关闭并立即拒绝构图。
- 处理：物理剪枝函数外层保持梯度开启，只在剪枝完成后的重建输出对照局部使用
  `torch.inference_mode()`。
- 防复发：checkpoint 级测试实际调用默认 Torch-Pruning 后端，不用 mock 绕过构图。

### 内存中能运行的剪枝图不一定可交付

- 风险：DepGraph 会修改普通模块属性；若目标 YAML 与结果不一致，重启进程后可能无法加载。
- 处理：保存前从目标配置重新创建模型、`strict=True` 加载全部权重，并要求随机输入输出
  逐元素完全相同；伴生 manifest 记录三份源文件哈希和剪枝统计。
- 交叉验证：同一小模型上，手工重建与 Torch-Pruning 的全部权重逐元素一致。

## 2026-07-26：ONNX 交付证据

### 导出样例没有覆盖相机嵌入的负值范围

- 现象：旧样例把全部 16 通道生成为 `[0,1]`，但相机嵌入契约允许 `[-1,1]`。
- 处理：通道 8–11 显式映射到 `[-1,1]`，有效 mask 仍固定为 1。

### ONNX 直接写正式路径会留下半截产物

- 风险：导出、Checker 或 ORT 对照中断时，目标文件已经可见，后续脚本可能误拾取。
- 处理：先写 `.tmp`，完成 Checker、数值对照、算子审计与 SHA256 后再原子替换；
  manifest 同样原子写入。

### PyTorch 已弃用旧 TorchScript ONNX 导出路径

- 现象：PyTorch 2.13 对 `dynamo=False` 发出弃用警告。
- 决策：当前继续显式锁定并写入 manifest，因为新导出器会改变算子分解；迁移必须以
  目标 HiAI CANN DDK 的 A/B 转换、ORT 数值和真机落点为证据，不能只追随默认值。

## 2026-07-26：商用级 Gate 自动化

### 文档表格无法阻止“未测试即通过”

- 现象：画质、算子、时延、内存、稳定与回退要求已有文档，但没有统一机器入口；
  缺失字段、总体平均值或错配产物仍可能被人工误判。
- 处理：新增 `check-release` 三态 Gate。证据不足为 `BLOCKED`，真实越界为 `FAIL`，
  只有逐域画质与全部端侧证据通过才为 `PASS`。
- 防复发：Gate 核验输入 YAML、OM 和交付文档 SHA256；单测覆盖完整通过、单域失败
  不可被平均抵消、缺 DDK 版本必须阻塞三种路径。

## 2026-07-27：SIDD RAW 训练数据选择性获取

### 官网 Small 链接与第三方镜像不满足 RAW 训练

- 现象：官网 SIDD Small/Medium 下载入口仍指向不可达的旧主机；可访问的 Kaggle
  镜像只有 `SIDD_Small_sRGB_Only` PNG。
- 风险：为了省事改用 sRGB 会把 ISP 去马赛克、降噪与色调映射结果混入标签，无法
  训练本项目要求的 RAW 域模块。
- 处理：拒绝该 sRGB 镜像，改用官网逐场景表中的 Codalab Raw-RGB Mirror 2。

### 为一帧下载 11.7 GB 整包不可持续

- 现象：实测一个场景 noisy ZIP 为 3,186,321,585 字节，GT ZIP 为
  8,506,719,167 字节；目标 frame 010 的压缩成员合计仅约 78 MB。
- 处理：固定 `remotezip==0.12.3`，读取 ZIP 中央目录后只对目标压缩区间发 HTTP
  Range GET，首个样例节省约 99.3% 传输量。
- 完整性：要求精确成员名、ZIP CRC、解压大小和本地 SHA256 全部一致，写场景及集合
  收据；中断只留下可删除的 `.partial`。

### Codalab 的 HEAD 与 Range GET 行为不同

- 现象：预签名下载 URL 的 `HEAD` 返回 `SignatureDoesNotMatch`，但同一 URL 的
  `GET Range` 正常。
- 处理：RemoteZip 启用 `support_suffix_range=True`，直接用后缀 Range 读取中央目录，
  不把 HEAD 失败误判为数据不可用。

### 跨表格行正则把 0013 URL 错配到 0014

- 现象：最初一次性正则跨越 HTML `<tr>` 边界，把下一行 URL 绑定给当前场景。
- 保护效果：下载器在中央目录中要求唯一存在 `0013_*_RAW_010.MAT`，发现归档实际是
  0014 后在创建数据文件前失败，没有形成错误配对。
- 处理：来源解析改为先逐 `<tr>` 隔离，再读取该行的 Raw-RGB Mirror 2；配置保存
  完整场景名，批量入口在联网前校验重复项和 held-out 名单。
- 防复发：URL 可访问并不代表内容属于目标场景，成员身份检查不可省略。

### 只按相机和高 ISO 选样会缺验证集

- 现象：首批五个实例覆盖五款相机，但物理 `scene_id` 只有 001/002/003；默认稳定
  哈希分别落入 test/train/train，没有任何 val。
- 处理：增加非 held-out 的 `scene_id=010` 作为 val 锚点。后续扩充必须审计物理
  scene 分布，实例数或相机数不能代替切分覆盖。

### 当前 PowerShell 的 ArgumentList API 不可用

- 现象：本机 `ProcessStartInfo.ArgumentList` 为 null，调用 `Add` 后仍启动了一个
  无参数隐藏 Python 进程。
- 处理：立即终止该精确 PID，改用不含空格的相对参数和 `Arguments` 字符串启动。
- 防复发：后台启动必须同时检查 PowerShell 错误流、实际 PID 和目标 `.partial`；
  打印出 PID 不等于业务命令已正确运行。
