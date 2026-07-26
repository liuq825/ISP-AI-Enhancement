#ifndef ISP_AI_ENHANCEMENT_H_
#define ISP_AI_ENHANCEMENT_H_

#include <cstddef>
#include <cstdint>

namespace isp_ai {

enum class Status : std::int32_t {
  kOk = 0,
  kInvalidInput = 1,
  kContractMismatch = 2,
  kCalibrationMismatch = 3,
  kModelLoadFailed = 4,
  kNpuTimeout = 5,
  kThermalFallback = 6,
  kNonFiniteOutput = 7,
  kRuntimeFailure = 8,
};

enum class CfaPattern : std::uint8_t { kRggb, kGrbg, kGbrg, kBggr };
enum class Mode : std::uint8_t { kSingle, kHdr, kMfnr };

struct EnhanceRequest {
  const std::uint16_t* raw = nullptr;
  std::size_t raw_stride_bytes = 0;
  std::int32_t raw_width = 0;
  std::int32_t raw_height = 0;
  std::int32_t bit_depth = 0;
  const char* sensor_id = nullptr;
  CfaPattern cfa = CfaPattern::kRggb;
  float black_level[4] = {};
  float white_level = 0.0F;
  float analog_gain = 1.0F;
  float digital_gain = 1.0F;
  float exposure_time_seconds = 0.0F;
  float wb_rg = 1.0F;
  float wb_bg = 1.0F;
  Mode mode = Mode::kSingle;
  const float* fusion_confidence = nullptr;
  const float* motion_ghost = nullptr;
  const char* calibration_version = nullptr;
  const char* model_version = nullptr;
  std::int32_t timeout_ms = 0;
};

struct EnhanceResult {
  std::uint16_t* enhanced_raw = nullptr;
  std::size_t output_stride_bytes = 0;
  float runtime_ms = 0.0F;
  float peak_memory_mb = 0.0F;
  std::int32_t tile_count = 0;
  Status status = Status::kRuntimeFailure;
  const char* fallback_reason = nullptr;
};

class Enhancer {
 public:
  virtual ~Enhancer() = default;
  virtual Status Initialize(const char* model_path, const char* manifest_path) = 0;
  virtual Status Enhance(const EnhanceRequest& request, EnhanceResult* result) = 0;
  virtual void Shutdown() = 0;
};

}  // namespace isp_ai

#endif  // ISP_AI_ENHANCEMENT_H_
