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
