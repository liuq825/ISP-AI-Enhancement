#ifndef ISP_AI_ENHANCEMENT_H_
#define ISP_AI_ENHANCEMENT_H_

#include <cstddef>
#include <cstdint>

namespace isp_ai {

/// AI Enhancement 接口状态码；调用方必须显式处理非 kOk 状态并执行回退。
enum class Status : std::int32_t {
  kOk = 0,                   ///< 端侧增强成功且输出通过完整性检查。
  kInvalidInput = 1,         ///< RAW 指针、尺寸、步幅或数值范围无效。
  kContractMismatch = 2,     ///< 运行时输入契约与模型清单不一致。
  kCalibrationMismatch = 3,  ///< 传感器标定版本与模型要求不一致。
  kModelLoadFailed = 4,      ///< 离线模型无法加载或完整性校验失败。
  kNpuTimeout = 5,           ///< NPU 执行超过调用方给定的超时时间。
  kThermalFallback = 6,      ///< 因温控策略主动跳过模型并回退传统 ISP。
  kNonFiniteOutput = 7,      ///< 模型产生 NaN/Inf，输出不得继续进入 ISP。
  kRuntimeFailure = 8,       ///< 其他运行时故障。
};

/// 传感器 Bayer 彩色滤光阵列在左上 2×2 像素中的排列。
enum class CfaPattern : std::uint8_t { kRggb, kGrbg, kGbrg, kBggr };

/// 拍摄模式决定融合置信度和运动鬼影条件通道的解释方式。
enum class Mode : std::uint8_t { kSingle, kHdr, kMfnr };

/// 一次增强请求的只读 RAW 缓冲区、标定信息和运行约束。
///
/// raw 指向未打包的 Bayer uint16 平面；实现负责依据 black_level、
/// white_level 和 cfa 构建规范的四通道浮点 RAW。所有字符串在 Enhance
/// 返回前必须保持有效，空间条件图尺寸需与打包后的 RAW 尺寸一致。
struct EnhanceRequest {
  const std::uint16_t* raw = nullptr;     ///< 输入 Bayer 首地址，不转移所有权。
  std::size_t raw_stride_bytes = 0;       ///< 相邻 RAW 行的字节步幅。
  std::int32_t raw_width = 0;             ///< 未打包 RAW 宽度，必须为偶数。
  std::int32_t raw_height = 0;            ///< 未打包 RAW 高度，必须为偶数。
  std::int32_t bit_depth = 0;             ///< 有效位深，用于输入范围校验。
  const char* sensor_id = nullptr;        ///< 对应相机嵌入注册表的稳定传感器 ID。
  CfaPattern cfa = CfaPattern::kRggb;     ///< 当前传感器 CFA 排列。
  float black_level[4] = {};              ///< 按 R/Gr/Gb/B 顺序的黑电平。
  float white_level = 0.0F;               ///< 线性 RAW 饱和值。
  float analog_gain = 1.0F;               ///< 模拟增益，必须为正数。
  float digital_gain = 1.0F;              ///< 数字增益，必须为正数。
  float exposure_time_seconds = 0.0F;     ///< 曝光秒数，用于曝光条件编码。
  float wb_rg = 1.0F;                     ///< R 相对 G 的白平衡比。
  float wb_bg = 1.0F;                     ///< B 相对 G 的白平衡比。
  Mode mode = Mode::kSingle;              ///< 单帧、HDR 或多帧降噪模式。
  const float* fusion_confidence = nullptr;  ///< 可选的打包分辨率融合置信图。
  const float* motion_ghost = nullptr;       ///< 可选的打包分辨率运动鬼影图。
  const char* calibration_version = nullptr; ///< 传感器标定版本。
  const char* model_version = nullptr;       ///< 调用方期望的模型版本。
  std::int32_t timeout_ms = 0;               ///< NPU 执行超时；非正值使用产品默认值。
};

/// 调用方提供的输出缓冲区及本次执行的可观测指标。
struct EnhanceResult {
  std::uint16_t* enhanced_raw = nullptr;  ///< 输出 Bayer 缓冲区，不转移所有权。
  std::size_t output_stride_bytes = 0;    ///< 输出相邻行的字节步幅。
  float runtime_ms = 0.0F;                ///< 整个增强调用的端到端耗时。
  float peak_memory_mb = 0.0F;            ///< 本次调用观察到的峰值内存。
  std::int32_t tile_count = 0;            ///< 分块数；整帧执行时为 1。
  Status status = Status::kRuntimeFailure; ///< 最终状态，失败时输出不得使用。
  const char* fallback_reason = nullptr;   ///< 可选回退原因，由实现管理生命周期。
};

/// 平台适配层的抽象接口；具体实现可绑定 HiAI 或 MindSpore Lite 运行时。
class Enhancer {
 public:
  virtual ~Enhancer() = default;

  /// 加载离线模型和模型清单，并核验二者的契约与完整性。
  virtual Status Initialize(const char* model_path, const char* manifest_path) = 0;

  /// 执行一次同步增强；任何失败都必须设置 result->status。
  virtual Status Enhance(const EnhanceRequest& request, EnhanceResult* result) = 0;

  /// 释放 NPU、模型和工作区资源；允许在失败初始化后安全调用。
  virtual void Shutdown() = 0;
};

}  // namespace isp_ai

#endif  // ISP_AI_ENHANCEMENT_H_
