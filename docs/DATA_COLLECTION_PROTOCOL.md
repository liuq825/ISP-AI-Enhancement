# 产品 RAW 数据采集协议

## 每颗 Sensor 的最低覆盖

- 标定：暗场、均匀光、灰阶、色卡、斜边、平场、坏点和镜头阴影。
- 场景：日光、室内、低照、极暗；人脸、毛发、文字、织物、建筑线、天空和霓虹。
- 增益：低、中、高、极高 ISO，并记录模拟/数字增益拆分。
- 设备：至少三个批次，多台设备，常温与高温。
- 模式：单帧、HDR 静态/运动、MFNR/夜景；保留 fusion confidence 与 ghost mask。

## 配对要求

1. 单帧 GT 使用长曝光低增益或高质量多帧融合，并统一曝光。
2. 执行亚像素对齐；对运动或无法可靠配对区域生成 valid/confidence mask。
3. 不把裁剪、压缩或 ISP 后图像伪装成线性 RAW GT。
4. 保存原始 DNG/RAW 和不可变元数据；转换产物记录脚本版本与 SHA256。

## 元数据最低字段

`device_id, sensor_id, lens_id, session_id, burst_id, scene_id, frame_id,
CFA, bit_depth, black_level[4], white_level, analog_gain, digital_gain,
exposure_time, ISO, WB, CCT, focus_position, temperature, mode, firmware`

## Golden Set

Golden Set 只增不改，并与训练数据物理隔离。每个 Sensor×模式×ISO 桶至少有独立
样本，统计均值、P10 和最差样本；另建 500-1000 组盲评集观察 banding、色偏、
涂抹、振铃、鬼影和 Tile 接缝。
