#include "hipcv/gpu_mat.hpp"

#include <limits>
#include <utility>

#if HIPCV_HAS_HIP
#include <hip/hip_runtime.h>
#endif

namespace hipcv {
namespace {

Status validate_shape(const ImageShape& shape) noexcept
{
    if (shape.width <= 0 || shape.height <= 0 || shape.channels <= 0) {
        return {StatusCode::invalid_argument, "image dimensions and channels must be positive"};
    }

    const auto min_step = static_cast<std::size_t>(shape.width) * static_cast<std::size_t>(shape.channels);
    if (shape.step_bytes != 0 && shape.step_bytes < min_step) {
        return {StatusCode::invalid_argument, "step_bytes is smaller than a packed row"};
    }

    return Status::success();
}

std::size_t normalized_step_bytes(const ImageShape& shape) noexcept
{
    if (shape.step_bytes != 0) {
        return shape.step_bytes;
    }

    return static_cast<std::size_t>(shape.width) * static_cast<std::size_t>(shape.channels);
}

bool checked_image_size(const ImageShape& shape, std::size_t* bytes) noexcept
{
    const auto step = normalized_step_bytes(shape);
    const auto height = static_cast<std::size_t>(shape.height);

    if (height != 0 && step > std::numeric_limits<std::size_t>::max() / height) {
        return false;
    }

    *bytes = step * height;
    return true;
}

} // namespace

GpuMat::~GpuMat()
{
    release();
}

GpuMat::GpuMat(GpuMat&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      shape_(std::exchange(other.shape_, ImageShape{})),
      size_bytes_(std::exchange(other.size_bytes_, 0))
{
}

GpuMat& GpuMat::operator=(GpuMat&& other) noexcept
{
    if (this != &other) {
        release();
        data_ = std::exchange(other.data_, nullptr);
        shape_ = std::exchange(other.shape_, ImageShape{});
        size_bytes_ = std::exchange(other.size_bytes_, 0);
    }

    return *this;
}

Status GpuMat::allocate(ImageShape shape) noexcept
{
    if (auto status = validate_shape(shape); !status.ok()) {
        return status;
    }

    shape.step_bytes = normalized_step_bytes(shape);

    std::size_t bytes = 0;
    if (!checked_image_size(shape, &bytes)) {
        return {StatusCode::invalid_argument, "image allocation size overflow"};
    }

#if HIPCV_HAS_HIP
    release();

    void* device_ptr = nullptr;
    if (hipMalloc(&device_ptr, bytes) != hipSuccess) {
        return {StatusCode::allocation_failed, "hipMalloc failed"};
    }

    data_ = device_ptr;
    shape_ = shape;
    size_bytes_ = bytes;
    return Status::success();
#else
    (void)bytes;
    return {StatusCode::hip_not_enabled, "hipcv was built without HIP support"};
#endif
}

Status GpuMat::upload(const void* host_data, ImageShape shape) noexcept
{
    if (host_data == nullptr) {
        return {StatusCode::invalid_argument, "host_data must not be null"};
    }

    if (auto status = allocate(shape); !status.ok()) {
        return status;
    }

#if HIPCV_HAS_HIP
    if (hipMemcpy(data_, host_data, size_bytes_, hipMemcpyHostToDevice) != hipSuccess) {
        release();
        return {StatusCode::copy_failed, "host-to-device hipMemcpy failed"};
    }

    return Status::success();
#else
    return {StatusCode::hip_not_enabled, "hipcv was built without HIP support"};
#endif
}

Status GpuMat::download(void* host_data, std::size_t host_capacity_bytes) const noexcept
{
    if (host_data == nullptr) {
        return {StatusCode::invalid_argument, "host_data must not be null"};
    }

    if (empty()) {
        return {StatusCode::invalid_argument, "GpuMat is empty"};
    }

    if (host_capacity_bytes < size_bytes_) {
        return {StatusCode::invalid_argument, "host buffer is too small"};
    }

#if HIPCV_HAS_HIP
    if (hipMemcpy(host_data, data_, size_bytes_, hipMemcpyDeviceToHost) != hipSuccess) {
        return {StatusCode::copy_failed, "device-to-host hipMemcpy failed"};
    }

    return Status::success();
#else
    return {StatusCode::hip_not_enabled, "hipcv was built without HIP support"};
#endif
}

void GpuMat::release() noexcept
{
#if HIPCV_HAS_HIP
    if (data_ != nullptr) {
        hipFree(data_);
    }
#endif

    data_ = nullptr;
    shape_ = {};
    size_bytes_ = 0;
}

} // namespace hipcv
