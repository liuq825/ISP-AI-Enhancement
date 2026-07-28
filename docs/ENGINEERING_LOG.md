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

### “真实数据能训练”不等于“数据足够商用品质”

- 风险：六对 RAW 已能跑通真实链路，但若直接复用正式 Student 配置，会生成看似完整
  的 checkpoint，容易被误当成 P0 基线。
- 处理：训练前门禁增加 train/val 最少样本、物理场景组、Sensor、ISO 和模式覆盖。
  选择性子集使用明确命名的 smoke 配置；公开 P0 配置以 SIDD Medium 的 320 个
  全分辨率源配对为基准，要求至少 3,200 个训练 patch、200 个独立训练源配对、
  64 个验证 patch、4 个独立验证源配对和五款相机/四个 ISO 桶；当前不足时在创建
  输出目录前拒绝。
- 修正：曾把每轮随机 crop 的数量误当成独立源记录，写出 10,000/300 门槛；这会让
  官方 Medium 永远无法通过。数据充分性必须区分“独立拍摄配对”和“从同一图采样的
  patch”，后者不能增加场景独立性。

### Mirror 2 批量清单比复制 160 行链接更可靠

- 发现：官网提供 72,800 字节的 `SIDD_URLs_Mirror_2.txt`，按每个非 held-out 场景
  固定列出 noisy RAW、GT RAW、noisy sRGB、GT sRGB、metadata 五个 URL。
- 处理：生成器逐 HTML `<tr>` 提取场景，确认 160 个训练/40 个 held-out，再要求
  URL 数严格为 800；每五个只取前两个 RAW URL。两份来源内容 SHA256 写入配置。
- 防复发：测试故意删除一个 URL，必须整体失败；不能让后续场景 URL 向前错位。

### 双帧不能重复打开同一个数 GB 归档

- 现象：若循环调用单帧入口，320 对会打开 640 个 noisy/GT 归档；同一场景的两个帧
  重复读取中央目录并重复执行重定向。
- 处理：多帧入口每场景只打开一次 noisy 和一次 GT，在同一上下文中提取 010/020；
  单元测试记录工厂调用次数，双帧必须仍只有两次打开。

### 160 场景任务不能靠人工反复重启

- 风险：公共镜像任一连接重置都可能让数小时任务退出；仅有断点复用仍需要人工值守。
- 处理：场景级默认最多尝试 4 次并线性退避 5/10/15 秒，失败事件写进度日志。已完成
  成员在下一次尝试中复核复用。
- 边界：成员缺失、held-out、大小/CRC 不一致均为确定性 `ValueError`，立即失败且
  不重试，防止把数据错配伪装成网络抖动。单测分别覆盖两条分支。

### 单一 CodaLab 镜像会让有界重试持续失败

- 现象：前四个场景完成后，CodaLab 对第五个场景连续返回代理断连；把场景重试从
  4 次增加到 12 次仍只是在同一故障端点等待。
- 探测：SIDD 官网同时发布 Mirror 1 `SIDD_URLs.txt`，同样含 800 个按五角色排列的
  URL。对第五场景仅请求 1 字节，服务器返回 `206 Partial Content`、
  `Accept-Ranges: bytes`，证明无需下载约 4.27 GB 整包即可继续选帧。
- 处理：配置生成器分别校验两份官方清单的数量、唯一性和 SHA256，为每个场景绑定
  一对主 URL 和一对备用 URL。主源出现 I/O 异常后同一轮立即切换备用源；两者都失败
  才进入线性退避。
- 时限：RemoteZip 使用 `(20, 120)` 秒的连接/读取二元超时。失效主源不会让每个场景
  多等待两分钟，正常的大成员流式读取仍可容忍 120 秒的单次 socket 停顿。
- 持续故障：显式 `--prefer-fallback` 只交换本次主/备尝试顺序，不重写版本化配置。
  默认仍优先 HTTPS；操作者确认主源连续故障后可避免每场景重复等待连接超时。
- 安全边界：身份、成员、CRC 或现有文件不一致仍是确定性错误，不允许靠切换镜像
  绕过。收据保存实际来源，离线复核只接受当前配置明确列出的完整 URL 对。

### 全分辨率压缩 NPZ 会拖慢随机裁剪

- 风险：若每个 DataLoader 样本只取 256 crop，却先解压约 50 MB 全图，CPU/磁盘会
  成为训练瓶颈；多 worker 还会放大瞬时内存。
- 处理：导入阶段每个源配对只加载和 CFA 打包一次，再确定性抽取 16 个不重复
  `256×256` packed RAW patch。训练时直接读取小 NPZ。
- 治理：Manifest 保留 `source_pair_id` 与坐标；门禁同时要求 patch 记录数和唯一源
  配对数，不能靠增加同图裁剪数量绕过独立数据下限。
- 门槛校正：320 对按物理场景稳定切分后为 train 206 / val 6 / test 108 个源配对，
  每对 16 patch 后为 3,296 / 96 / 1,728 条记录。因此正式门槛设为 train
  `>=3,200 records + >=200 source pairs`、val `>=64 + >=4`，而非不可达的 250。

### 断点重启不应重新访问所有已完成 ZIP

- 现象：已有 pair 收据和 MAT 完整，但旧恢复仍从场景 1 打开远端中央目录，完成越多，
  每次重启的无效请求越多。
- 处理：同场景全部收据存在时，核对当前配置 URL、场景/帧身份，并重算本地文件
  SHA256/CRC/大小；全部通过即离线复用，不访问网络。
- 安全边界：收据不齐时进入原远程恢复；收据齐但本地文件被篡改时硬失败，不自动用
  网络内容覆盖证据。测试用禁止调用的归档工厂证明离线路径没有网络依赖。

### 上万个 NPZ 文件的 patch 导入也必须可恢复

- 风险：320 源对×16 patch 会形成 5,120 条 Manifest 记录，并写 10,240 个
  input/target NPZ 文件；末尾中断后若从头重新压缩，浪费数小时，并可能覆盖不同
  seed 的旧产物。
- 处理：目标 NPZ 已存在时只允许单一 `raw` 字段、float32 且与当前确定性转换逐元素
  完全相同；满足时保留原文件和 mtime，否则硬失败。临时文件仍原子替换。
- 防复发：测试对同一输出目录重复 patch 导入，要求 Manifest 相同且已完成 NPZ 的
  纳秒 mtime 不变。

### 最终数据收据不能靠手工复制 320 行哈希

- 风险：状态命令为快速监控只检查收据字段与文件大小；人工从 320 个 pair 收据汇总
  SHA256 容易漏行、错帧，也无法证明集合收据仍对应当前配置。
- 处理：`audit-sidd-subset` 只接受完整集合，重新计算 640 个 MAT 的 SHA256/CRC32，
  核对集合收据、逐配对收据、配置 SHA 和实际主/备 URL，再原子写可提交 Git 的 YAML。
- 防复发：测试先生成有效审计收据，再等长改写一个 MAT；大小不变也必须因 SHA/CRC
  不一致而拒绝，证明最终审计不是只看文件长度。

### 官方 0032 ZIP 把实例号从 basename 移到了父目录

- 现象：Mirror 1 的 `0032_NOISY_RAW.zip` 含 150 个成员，frame 010/020 均存在，
  但文件名是 `NOISY_RAW_010.MAT`，实例号只出现在两层 `0032_NOISY_RAW` 父目录；
  GT 同样使用 `0032_GT_RAW/.../GT_RAW_010.MAT`。只按此前的
  `0032_NOISY_RAW_010.MAT` basename 查找会把真实官方归档误报为缺成员。
- 处理：成员查找兼容“实例号在 basename”和“实例号在父目录”两种官方布局。
- 安全边界：通用 basename 只有在父目录组件精确等于当前实例+角色令牌时才匹配；
  测试把 `NOISY_RAW_010.MAT` 放进 `0033_NOISY_RAW`，对 0032 请求必须拒绝，不能
  为兼容格式而放松场景身份。

### 320 对实际获取结果

- 完成：160 个非 held-out 场景、frame 010/020，共 320 对、640 个 MAT，
  `21,618,970,008` 字节；最终状态为 100%，无错误、无 `.partial`。
- 来源：最初 8 对来自 CodaLab，持续故障后 312 对来自官网列出的 Mirror 1；逐 pair
  收据保留实际 URL，不能把来源切换隐藏在汇总数字中。
- 最终复核：重新读取全部 MAT 并核对 CRC32/SHA256、配置、集合收据和 320 个 pair
  收据。可提交审计收据为 `resources/sidd_medium_receipt.yaml`，详细报告见
  `docs/SIDD_MEDIUM_ACQUISITION.md`。

### Windows 环境同时存在 Path/PATH 会让 Start-Process 在业务代码前失败

- 现象：尝试把 5,120 条 patch 导入放到隐藏后台进程时，`Start-Process` 报告
  “已添加项，字典中的关键字 Path/PATH”，加 `-UseNewEnvironment` 仍失败。
- 判断：目标目录尚未创建，且没有业务 Python 进程，说明异常发生在 PowerShell
  构建子进程环境阶段，不是 SIDD 导入器崩溃；不能看到启动命令就假设任务在执行。
- 处理：改为直接以前台 `.venv\Scripts\python.exe -m isp_ai_enhancement.cli`
  运行同一命令，导入在 998.3 秒后正常完成。失败尝试没有留下半成品。
- 防复发：后台任务启动后同时检查错误流、实际命令行和首个业务产物；受影响主机优先
  前台执行，若必须后台运行则先用最小显式环境构造子进程，不能依赖继承的重复键。

### 数量正确仍不足以证明上万个 NPZ 可训练

- 风险：Manifest 有 5,120 行、目录有 10,240 个文件，只能证明路径数量对得上；
  NPZ 可能含对象数组、额外字段、错误 dtype、NaN、越界 RAW 或 noisy/GT shape 错配。
- 处理：新增 `audit-sidd-import`，逐个以 `allow_pickle=False` 打开 NPZ，要求唯一
  `raw`、float32、`4×H×W`、有限且位于 `[0,1]`，同时核对 patch 元数据、源 SHA、
  文件唯一引用、input/target shape 和正式训练数据门槛。
- 可移植摘要：压缩 NPZ 的 ZIP 时间戳可能导致文件 SHA 跨重建变化，因此按相对路径、
  dtype、shape 和解压数组字节生成内容摘要；数值相同即得到相同摘要。
- 实际结果：5,120 条记录、10,240 个 NPZ 全部通过；压缩文件
  `6,155,310,343` 字节，解压数组 `10,737,418,240` 字节，内容 SHA256 为
  `d9dba6d6ac422d8cc0cf20f9b69f9bd21e019f4c5b5925d543b360d47dae3876`。

## 2026-07-28：Student、结构感知剪枝与联合蒸馏升级

### Student enc3 加深后所有参数基线必须重算

- 变更：Student 从 `encoder_blocks=[2,2,4,8]` 升级为 `[2,2,6,8]`，增加两个
  中尺度 enc3 block；模型仍由 YAML 驱动，不能在训练或部署脚本中散落块数常量。
- 影响：P0 参数从历史 `14,348,516` 变为 `14,586,340`。旧剪枝目标的 enc3 宽度
  数量与新拓扑不匹配，会在模型构造期被严格拒绝，不能直接沿用旧 checkpoint。
- 防复发：模型测试同时锁定 `encoder_blocks` 和精确参数数；剪枝目标 YAML 必须和
  P0 的九个 stage block 数一致。

### 全局 15% 不能被误解为每个 block 统一缩 15%

- 风险：统一比例会同时伤害高分辨率浅层、stage 边界和 skip 接口，却可能仅凭总体
  参数率“达标”；总数正确不能证明结构合理。
- 处理：enc1/enc2/dec3/dec4 完整保留，enc3/enc4 首尾 block 比内部更宽，Middle
  和深层 Decoder 承担更多预算。当前图为 `14,586,340 → 12,405,108`，
  物理剪枝率 `14.953936%`。
- 可审计性：CLI 和 checkpoint manifest 增加 `stage_hidden_retention`；当前
  enc1/enc2 为 100%、enc4 为 87.5%、Middle 为 82.03125%，明确证明不是统一模板。
- 边界：这仍是结构先验起点。正式通道数必须用收敛 P0 做逐 block 短微调和逐域敏感度
  回退；随机权重的重要性与静态宽度表都不能作为商用品质结论。

### 只有输出蒸馏会把监督集中在图尾

- 现状：工程已经有多尺度 feature 投影，但缺少空间 attention 迁移，配置仍容易被
  概括为“Teacher 输出蒸馏”。
- 处理：输出权重从 0.30 降为 0.10 辅助项，新增 0.15 feature 和 0.10 attention。
  attention 对每层通道均方能量做逐样本 L2 归一化，不要求 Teacher/Student 通道一致。
- 恢复坑：attention key 没有天然参数，若只保存投影权重，改变层位后
  `strict=True` 也可能不报错。实现为每个 attention key 注册持久语义 buffer，使层位
  变化必然触发 checkpoint 状态不匹配。
- 验证：单测证明联合损失只向 Student/投影层传播梯度；极小训练闭环的 history 同时
  出现 output、feature、attention 三项，checkpoint 保存 distiller 语义状态。

### 有 distill.yaml 但没有 Teacher 训练入口不能称为可复现

- 现象：蒸馏配置引用 `checkpoints/teacher.pt`，仓库却没有生成该文件的正式配置，
  新开发者必须猜测训练轮数、数据门槛和复制步骤。
- 处理：新增 `configs/train_teacher.yaml`，输出直接落到 `runs/teacher/best.pt`；
  `distill.yaml` 直接引用该路径。完整顺序写入 `docs/REPRODUCTION_GUIDE.md`。
- 显存策略：Teacher+Student 同时前向的 batch 8 风险过高，正式 Teacher、蒸馏、
  剪枝微调和 QAT 使用 micro-batch 1、8 步梯度累积。最后不足 8 步的分组按实际
  micro-batch 数归一化，`global_step` 只在 optimizer 更新后增加。
- 防复发：复现指南中的每个输入 checkpoint 必须能由前一条版本化命令直接产生，
  不允许依赖未记录的手工复制、重命名或个人目录。

## 2026-07-28：CPU QAT 全链路演练

### 本机 QAT 跑通不等于目标 NPU 已通过

- 处理：新增 `train_local_fp32_smoke.yaml`、`train_local_qat_smoke.yaml` 与
  `export_local_qat_smoke.yaml`，从新生成的合成 RAW 依次执行 FP32、QAT、Q/DQ ONNX
  和 ONNX 审计；运行产物保留在被 Git 忽略的 `data/`、`runs/` 与 `artifacts/`。
- 实际结果：QAT checkpoint 可加载，导出静态 `1×16×64×64` 图，包含 124 个
  QuantizeLinear 与 124 个 DequantizeLinear；Checker 和 ORT 对照通过。
- 边界：该模型仅有 291,724 参数、合成数据训练一轮；它只能证明部署前工程链路，不能
  支持真实画质、麒麟 9000 时延、功耗或 OM 兼容性结论。

### QAT 导出误差必须按输入规格审计，不能全局放宽阈值

- 现象：同一 QAT 烟雾权重在 `512×512` ONNX/ORT 对照中出现最大绝对误差
  `2.156e-3`，超过正式导出配置的 `1e-4` 绝对容差；在本地演练使用的 `64×64` 静态图
  中误差为 `3.28e-7`。
- 处理：只为本地烟雾模型增加独立 `export_local_qat_smoke.yaml`，保留正式
  `configs/export_onnx.yaml` 的严格阈值；模型 manifest 记录导出配置哈希和实测误差。
- 防复发：不能用“QAT 有舍入误差”作为放宽正式阈值的理由。任何分辨率、QAT 图或运行时
  变化，都必须重新执行 Checker、ORT 对照和目标 DDK/真机审计。
