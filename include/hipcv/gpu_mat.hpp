#pragma once

#include <cstddef>
#include <cstdint>

#include "hipcv/status.hpp"

namespace hipcv {

enum class PixelFormat {
    unknown,
    gray8,
    rgb8,
    bgr8,
    rgba8,
    bgra8
};

struct ImageShape {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::size_t step_bytes = 0;
    PixelFormat format = PixelFormat::unknown;
};

class GpuMat {
public:
    GpuMat() = default;
    ~GpuMat();

    GpuMat(const GpuMat&) = delete;
    GpuMat& operator=(const GpuMat&) = delete;

    GpuMat(GpuMat&& other) noexcept;
    GpuMat& operator=(GpuMat&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept
    {
        return data_ == nullptr || shape_.width <= 0 || shape_.height <= 0;
    }

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] ImageShape shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

    Status allocate(ImageShape shape) noexcept;
    Status upload(const void* host_data, ImageShape shape) noexcept;
    Status download(void* host_data, std::size_t host_capacity_bytes) const noexcept;
    Status copyTo(GpuMat& dst) const noexcept;
    void release() noexcept;

private:
    void* data_ = nullptr;
    ImageShape shape_{};
    std::size_t size_bytes_ = 0;
};

} // namespace hipcv
