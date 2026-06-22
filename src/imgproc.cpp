#include "hipcv/imgproc.hpp"

#include <cstddef>
#include <cstdint>

#if HIPCV_HAS_HIP
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <mutex>
#include <string>
#include <vector>
#endif

namespace hipcv {
namespace {

bool is_supported_color_conversion(const ImageShape& shape, ColorConversion code) noexcept
{
    if (shape.channels != 3) {
        return false;
    }

    switch (code) {
    case ColorConversion::bgr_to_gray:
        return shape.format == PixelFormat::bgr8;
    case ColorConversion::rgb_to_gray:
        return shape.format == PixelFormat::rgb8;
    }

    return false;
}

ImageShape gray_shape_from(const ImageShape& src) noexcept
{
    return {
        src.width,
        src.height,
        1,
        static_cast<std::size_t>(src.width),
        PixelFormat::gray8,
    };
}

#if HIPCV_HAS_HIP

constexpr const char* kCvtColorKernelSource = R"(
extern "C" __global__ void cvt_color_to_gray(
    const unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    unsigned long long src_step,
    unsigned long long dst_step,
    int red_index)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* pixel = src + (static_cast<unsigned long long>(y) * src_step) + (x * 3);
    const unsigned int r = pixel[red_index];
    const unsigned int g = pixel[1];
    const unsigned int b = pixel[2 - red_index];
    const unsigned int gray = (77u * r + 150u * g + 29u * b + 128u) >> 8;

    dst[(static_cast<unsigned long long>(y) * dst_step) + x] = static_cast<unsigned char>(gray);
}
)";

struct CvtColorKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;

    ~CvtColorKernel()
    {
        if (module != nullptr) {
            hipModuleUnload(module);
        }
    }
};

CvtColorKernel& cvt_color_kernel() noexcept
{
    static CvtColorKernel kernel;
    return kernel;
}

std::mutex& cvt_color_kernel_mutex() noexcept
{
    static std::mutex mutex;
    return mutex;
}

Status compile_cvt_color_kernel() noexcept
{
    auto& kernel = cvt_color_kernel();
    if (kernel.function != nullptr) {
        return Status::success();
    }

    std::lock_guard<std::mutex> lock(cvt_color_kernel_mutex());
    if (kernel.function != nullptr) {
        return Status::success();
    }

    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        return {StatusCode::runtime_error, "hipGetDevice failed"};
    }

    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess) {
        return {StatusCode::runtime_error, "hipGetDeviceProperties failed"};
    }

    const std::string architecture = std::string("--gpu-architecture=") + props.gcnArchName;
    const char* options[] = {architecture.c_str()};

    hiprtcProgram program{};
    if (hiprtcCreateProgram(&program, kCvtColorKernelSource, "cvt_color_to_gray.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
        return {StatusCode::runtime_error, "hiprtcCreateProgram failed"};
    }

    const auto compile_result = hiprtcCompileProgram(program, 1, options);
    if (compile_result != HIPRTC_SUCCESS) {
        hiprtcDestroyProgram(&program);
        return {StatusCode::runtime_error, "hiprtcCompileProgram failed"};
    }

    std::size_t code_size = 0;
    if (hiprtcGetCodeSize(program, &code_size) != HIPRTC_SUCCESS || code_size == 0) {
        hiprtcDestroyProgram(&program);
        return {StatusCode::runtime_error, "hiprtcGetCodeSize failed"};
    }

    std::vector<char> code(code_size);
    if (hiprtcGetCode(program, code.data()) != HIPRTC_SUCCESS) {
        hiprtcDestroyProgram(&program);
        return {StatusCode::runtime_error, "hiprtcGetCode failed"};
    }

    hiprtcDestroyProgram(&program);

    hipModule_t module = nullptr;
    if (hipModuleLoadData(&module, code.data()) != hipSuccess) {
        return {StatusCode::runtime_error, "hipModuleLoadData failed"};
    }

    hipFunction_t function = nullptr;
    if (hipModuleGetFunction(&function, module, "cvt_color_to_gray") != hipSuccess) {
        hipModuleUnload(module);
        return {StatusCode::runtime_error, "hipModuleGetFunction failed"};
    }

    kernel.module = module;
    kernel.function = function;
    return Status::success();
}

#endif

} // namespace

Status cvtColor(const GpuMat& src, GpuMat& dst, ColorConversion code) noexcept
{
    if (src.empty()) {
        return {StatusCode::invalid_argument, "source GpuMat is empty"};
    }

    const auto src_shape = src.shape();
    if (!is_supported_color_conversion(src_shape, code)) {
        return {StatusCode::invalid_argument, "unsupported color conversion"};
    }

    if (auto status = dst.allocate(gray_shape_from(src_shape)); !status.ok()) {
        return status;
    }

#if HIPCV_HAS_HIP
    try {
        if (auto status = compile_cvt_color_kernel(); !status.ok()) {
            dst.release();
            return status;
        }

        const auto dst_shape = dst.shape();
        const auto* src_ptr = static_cast<const std::uint8_t*>(src.data());
        auto* dst_ptr = static_cast<std::uint8_t*>(dst.data());
        const int width = src_shape.width;
        const int height = src_shape.height;
        const auto src_step = static_cast<unsigned long long>(src_shape.step_bytes);
        const auto dst_step = static_cast<unsigned long long>(dst_shape.step_bytes);
        const int red_index = code == ColorConversion::bgr_to_gray ? 2 : 0;

        void* args[] = {
            const_cast<std::uint8_t**>(&src_ptr),
            &dst_ptr,
            const_cast<int*>(&width),
            const_cast<int*>(&height),
            const_cast<unsigned long long*>(&src_step),
            const_cast<unsigned long long*>(&dst_step),
            const_cast<int*>(&red_index),
        };

        constexpr unsigned int block_x = 16;
        constexpr unsigned int block_y = 16;
        const auto grid_x = static_cast<unsigned int>((width + static_cast<int>(block_x) - 1) / static_cast<int>(block_x));
        const auto grid_y = static_cast<unsigned int>((height + static_cast<int>(block_y) - 1) / static_cast<int>(block_y));

        if (hipModuleLaunchKernel(cvt_color_kernel().function, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr) != hipSuccess) {
            dst.release();
            return {StatusCode::runtime_error, "hipModuleLaunchKernel failed"};
        }

        if (hipDeviceSynchronize() != hipSuccess) {
            dst.release();
            return {StatusCode::runtime_error, "hipDeviceSynchronize failed"};
        }

        return Status::success();
    } catch (...) {
        dst.release();
        return {StatusCode::runtime_error, "cvtColor failed"};
    }
#else
    return {StatusCode::hip_not_enabled, "hipcv was built without HIP support"};
#endif
}

} // namespace hipcv
