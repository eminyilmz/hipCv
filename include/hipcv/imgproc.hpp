#pragma once

#include "hipcv/gpu_mat.hpp"
#include "hipcv/status.hpp"

namespace hipcv {

enum class ColorConversion {
    bgr_to_gray,
    rgb_to_gray
};

Status cvtColor(const GpuMat& src, GpuMat& dst, ColorConversion code) noexcept;

} // namespace hipcv
