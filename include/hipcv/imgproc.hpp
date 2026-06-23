#pragma once

#include <cstdint>

#include "hipcv/gpu_mat.hpp"
#include "hipcv/status.hpp"

namespace hipcv {

enum class ColorConversion {
    bgr_to_gray,
    rgb_to_gray
};

enum class ResizeInterpolation {
    nearest
};

enum class ThresholdType {
    binary,
    binary_inv
};

Status cvtColor(const GpuMat& src, GpuMat& dst, ColorConversion code) noexcept;
Status resize(const GpuMat& src, GpuMat& dst, int width, int height, ResizeInterpolation interpolation = ResizeInterpolation::nearest) noexcept;
Status threshold(const GpuMat& src, GpuMat& dst, std::uint8_t threshold_value, std::uint8_t max_value, ThresholdType type = ThresholdType::binary) noexcept;
Status blur(const GpuMat& src, GpuMat& dst, int kernel_width, int kernel_height) noexcept;
Status gaussianBlur(const GpuMat& src, GpuMat& dst, int kernel_width, int kernel_height) noexcept;

} // namespace hipcv
