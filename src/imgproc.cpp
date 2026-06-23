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
    case ColorConversion::bgr_to_rgb:
        return shape.format == PixelFormat::bgr8;
    case ColorConversion::rgb_to_bgr:
        return shape.format == PixelFormat::rgb8;
    }

    return false;
}

ImageShape color_conversion_shape_from(const ImageShape& src, ColorConversion code) noexcept
{
    switch (code) {
    case ColorConversion::bgr_to_gray:
    case ColorConversion::rgb_to_gray:
        return {
            src.width,
            src.height,
            1,
            static_cast<std::size_t>(src.width),
            PixelFormat::gray8,
        };
    case ColorConversion::bgr_to_rgb:
        return {
            src.width,
            src.height,
            3,
            static_cast<std::size_t>(src.width) * 3u,
            PixelFormat::rgb8,
        };
    case ColorConversion::rgb_to_bgr:
        return {
            src.width,
            src.height,
            3,
            static_cast<std::size_t>(src.width) * 3u,
            PixelFormat::bgr8,
        };
    }

    return {};
}

bool is_supported_resize_shape(const ImageShape& shape) noexcept
{
    switch (shape.format) {
    case PixelFormat::gray8:
    case PixelFormat::rgb8:
    case PixelFormat::bgr8:
    case PixelFormat::rgba8:
    case PixelFormat::bgra8:
        return shape.channels == 1 || shape.channels == 3 || shape.channels == 4;
    case PixelFormat::unknown:
        return false;
    }

    return false;
}

ImageShape resized_shape_from(const ImageShape& src, int width, int height) noexcept
{
    return {
        width,
        height,
        src.channels,
        static_cast<std::size_t>(width) * static_cast<std::size_t>(src.channels),
        src.format,
    };
}

bool is_supported_threshold_shape(const ImageShape& shape) noexcept
{
    return shape.format == PixelFormat::gray8 && shape.channels == 1;
}

bool is_supported_blur_shape(const ImageShape& shape) noexcept
{
    switch (shape.format) {
    case PixelFormat::gray8:
        return shape.channels == 1;
    case PixelFormat::rgb8:
    case PixelFormat::bgr8:
        return shape.channels == 3;
    case PixelFormat::rgba8:
    case PixelFormat::bgra8:
        return shape.channels == 4;
    case PixelFormat::unknown:
        return false;
    }

    return false;
}

bool is_supported_gaussian_blur_shape(const ImageShape& shape) noexcept
{
    switch (shape.format) {
    case PixelFormat::gray8:
        return shape.channels == 1;
    case PixelFormat::rgb8:
    case PixelFormat::bgr8:
        return shape.channels == 3;
    case PixelFormat::rgba8:
    case PixelFormat::bgra8:
        return shape.channels == 4;
    case PixelFormat::unknown:
        return false;
    }

    return false;
}

ImageShape same_shape_as(const ImageShape& src) noexcept
{
    return {
        src.width,
        src.height,
        src.channels,
        src.step_bytes,
        src.format,
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

extern "C" __global__ void cvt_color_swap_rb(
    const unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    unsigned long long src_step,
    unsigned long long dst_step)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* src_pixel = src + (static_cast<unsigned long long>(y) * src_step) + (x * 3);
    unsigned char* dst_pixel = dst + (static_cast<unsigned long long>(y) * dst_step) + (x * 3);

    dst_pixel[0] = src_pixel[2];
    dst_pixel[1] = src_pixel[1];
    dst_pixel[2] = src_pixel[0];
}
)";

constexpr const char* kResizeNearestKernelSource = R"(
extern "C" __global__ void resize_nearest_u8(
    const unsigned char* src,
    unsigned char* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    unsigned long long src_step,
    unsigned long long dst_step)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    const int src_x = (x * src_width) / dst_width;
    const int src_y = (y * src_height) / dst_height;
    const unsigned char* src_pixel = src + (static_cast<unsigned long long>(src_y) * src_step) + (src_x * channels);
    unsigned char* dst_pixel = dst + (static_cast<unsigned long long>(y) * dst_step) + (x * channels);

    for (int channel = 0; channel < channels; ++channel) {
        dst_pixel[channel] = src_pixel[channel];
    }
}
)";

constexpr const char* kResizeBilinearKernelSource = R"(
extern "C" __global__ void resize_bilinear_u8(
    const unsigned char* src,
    unsigned char* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    unsigned long long src_step,
    unsigned long long dst_step)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
    const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
    float src_fx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
    float src_fy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

    if (src_fx < 0.0f) {
        src_fx = 0.0f;
    }
    if (src_fy < 0.0f) {
        src_fy = 0.0f;
    }

    int x0 = static_cast<int>(src_fx);
    int y0 = static_cast<int>(src_fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x1 >= src_width) {
        x1 = src_width - 1;
    }
    if (y1 >= src_height) {
        y1 = src_height - 1;
    }

    const float tx = src_fx - static_cast<float>(x0);
    const float ty = src_fy - static_cast<float>(y0);

    const unsigned char* row0 = src + (static_cast<unsigned long long>(y0) * src_step);
    const unsigned char* row1 = src + (static_cast<unsigned long long>(y1) * src_step);
    unsigned char* dst_pixel = dst + (static_cast<unsigned long long>(y) * dst_step) + (x * channels);

    for (int channel = 0; channel < channels; ++channel) {
        const float top_left = static_cast<float>(row0[(x0 * channels) + channel]);
        const float top_right = static_cast<float>(row0[(x1 * channels) + channel]);
        const float bottom_left = static_cast<float>(row1[(x0 * channels) + channel]);
        const float bottom_right = static_cast<float>(row1[(x1 * channels) + channel]);
        const float top = top_left + ((top_right - top_left) * tx);
        const float bottom = bottom_left + ((bottom_right - bottom_left) * tx);
        const float value = top + ((bottom - top) * ty);

        dst_pixel[channel] = static_cast<unsigned char>(value + 0.5f);
    }
}
)";

constexpr const char* kThresholdKernelSource = R"(
extern "C" __global__ void threshold_gray8(
    const unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    unsigned long long src_step,
    unsigned long long dst_step,
    unsigned int threshold_value,
    unsigned int max_value,
    int inverse)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const unsigned char value = src[(static_cast<unsigned long long>(y) * src_step) + x];
    const bool pass = value > threshold_value;
    const unsigned char out = (pass != (inverse != 0)) ? static_cast<unsigned char>(max_value) : 0;
    dst[(static_cast<unsigned long long>(y) * dst_step) + x] = out;
}
)";

constexpr const char* kBlurKernelSource = R"(
extern "C" __global__ void blur_u8(
    const unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    unsigned long long src_step,
    unsigned long long dst_step,
    int channels,
    int kernel_width,
    int kernel_height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int radius_x = kernel_width / 2;
    const int radius_y = kernel_height / 2;
    unsigned char* dst_pixel = dst + (static_cast<unsigned long long>(y) * dst_step) + (x * channels);

    for (int channel = 0; channel < channels; ++channel) {
        unsigned int sum = 0;

        for (int ky = 0; ky < kernel_height; ++ky) {
            int src_y = y + ky - radius_y;
            if (src_y < 0) {
                src_y = 0;
            } else if (src_y >= height) {
                src_y = height - 1;
            }

            for (int kx = 0; kx < kernel_width; ++kx) {
                int src_x = x + kx - radius_x;
                if (src_x < 0) {
                    src_x = 0;
                } else if (src_x >= width) {
                    src_x = width - 1;
                }

                const unsigned char* src_pixel = src + (static_cast<unsigned long long>(src_y) * src_step) + (src_x * channels);
                sum += src_pixel[channel];
            }
        }

        const unsigned int area = static_cast<unsigned int>(kernel_width * kernel_height);
        dst_pixel[channel] = static_cast<unsigned char>((sum + (area / 2u)) / area);
    }
}
)";

constexpr const char* kGaussianBlurKernelSource = R"(
__device__ int gaussian_weight(int offset, int kernel_size)
{
    const int absolute_offset = offset < 0 ? -offset : offset;
    if (kernel_size == 3) {
        return absolute_offset == 0 ? 2 : 1;
    }

    if (absolute_offset == 0) {
        return 6;
    }
    return absolute_offset == 1 ? 4 : 1;
}

extern "C" __global__ void gaussian_blur_u8(
    const unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    unsigned long long src_step,
    unsigned long long dst_step,
    int channels,
    int kernel_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int radius = kernel_size / 2;
    unsigned char* dst_pixel = dst + (static_cast<unsigned long long>(y) * dst_step) + (x * channels);

    for (int channel = 0; channel < channels; ++channel) {
        unsigned int sum = 0;

        for (int ky = -radius; ky <= radius; ++ky) {
            int src_y = y + ky;
            if (src_y < 0) {
                src_y = 0;
            } else if (src_y >= height) {
                src_y = height - 1;
            }

            const unsigned int weight_y = static_cast<unsigned int>(gaussian_weight(ky, kernel_size));
            for (int kx = -radius; kx <= radius; ++kx) {
                int src_x = x + kx;
                if (src_x < 0) {
                    src_x = 0;
                } else if (src_x >= width) {
                    src_x = width - 1;
                }

                const unsigned int weight_x = static_cast<unsigned int>(gaussian_weight(kx, kernel_size));
                const unsigned int weight = weight_x * weight_y;
                const unsigned char* src_pixel = src + (static_cast<unsigned long long>(src_y) * src_step) + (src_x * channels);
                sum += weight * static_cast<unsigned int>(src_pixel[channel]);
            }
        }

        const unsigned int weight_sum = kernel_size == 3 ? 16u : 256u;
        dst_pixel[channel] = static_cast<unsigned char>((sum + (weight_sum / 2u)) / weight_sum);
    }
}
)";

struct CvtColorKernel {
    hipModule_t module = nullptr;
    hipFunction_t gray_function = nullptr;
    hipFunction_t swap_rb_function = nullptr;

    ~CvtColorKernel()
    {
        if (module != nullptr) {
            hipModuleUnload(module);
        }
    }
};

struct ResizeNearestKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;

    ~ResizeNearestKernel()
    {
        if (module != nullptr) {
            hipModuleUnload(module);
        }
    }
};

struct ResizeBilinearKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;

    ~ResizeBilinearKernel()
    {
        if (module != nullptr) {
            hipModuleUnload(module);
        }
    }
};

struct ThresholdKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;

    ~ThresholdKernel()
    {
        if (module != nullptr) {
            hipModuleUnload(module);
        }
    }
};

struct BlurKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;

    ~BlurKernel()
    {
        if (module != nullptr) {
            hipModuleUnload(module);
        }
    }
};

struct GaussianBlurKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;

    ~GaussianBlurKernel()
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

ResizeNearestKernel& resize_nearest_kernel() noexcept
{
    static ResizeNearestKernel kernel;
    return kernel;
}

std::mutex& resize_nearest_kernel_mutex() noexcept
{
    static std::mutex mutex;
    return mutex;
}

ResizeBilinearKernel& resize_bilinear_kernel() noexcept
{
    static ResizeBilinearKernel kernel;
    return kernel;
}

std::mutex& resize_bilinear_kernel_mutex() noexcept
{
    static std::mutex mutex;
    return mutex;
}

ThresholdKernel& threshold_kernel() noexcept
{
    static ThresholdKernel kernel;
    return kernel;
}

std::mutex& threshold_kernel_mutex() noexcept
{
    static std::mutex mutex;
    return mutex;
}

BlurKernel& blur_kernel() noexcept
{
    static BlurKernel kernel;
    return kernel;
}

std::mutex& blur_kernel_mutex() noexcept
{
    static std::mutex mutex;
    return mutex;
}

GaussianBlurKernel& gaussian_blur_kernel() noexcept
{
    static GaussianBlurKernel kernel;
    return kernel;
}

std::mutex& gaussian_blur_kernel_mutex() noexcept
{
    static std::mutex mutex;
    return mutex;
}

Status compile_cvt_color_kernel() noexcept
{
    auto& kernel = cvt_color_kernel();
    if (kernel.gray_function != nullptr && kernel.swap_rb_function != nullptr) {
        return Status::success();
    }

    std::lock_guard<std::mutex> lock(cvt_color_kernel_mutex());
    if (kernel.gray_function != nullptr && kernel.swap_rb_function != nullptr) {
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

    hipFunction_t gray_function = nullptr;
    if (hipModuleGetFunction(&gray_function, module, "cvt_color_to_gray") != hipSuccess) {
        hipModuleUnload(module);
        return {StatusCode::runtime_error, "hipModuleGetFunction failed"};
    }

    hipFunction_t swap_rb_function = nullptr;
    if (hipModuleGetFunction(&swap_rb_function, module, "cvt_color_swap_rb") != hipSuccess) {
        hipModuleUnload(module);
        return {StatusCode::runtime_error, "hipModuleGetFunction failed"};
    }

    kernel.module = module;
    kernel.gray_function = gray_function;
    kernel.swap_rb_function = swap_rb_function;
    return Status::success();
}

Status compile_blur_kernel() noexcept
{
    auto& kernel = blur_kernel();
    if (kernel.function != nullptr) {
        return Status::success();
    }

    std::lock_guard<std::mutex> lock(blur_kernel_mutex());
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
    if (hiprtcCreateProgram(&program, kBlurKernelSource, "blur_u8.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
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
    if (hipModuleGetFunction(&function, module, "blur_u8") != hipSuccess) {
        hipModuleUnload(module);
        return {StatusCode::runtime_error, "hipModuleGetFunction failed"};
    }

    kernel.module = module;
    kernel.function = function;
    return Status::success();
}

Status compile_gaussian_blur_kernel() noexcept
{
    auto& kernel = gaussian_blur_kernel();
    if (kernel.function != nullptr) {
        return Status::success();
    }

    std::lock_guard<std::mutex> lock(gaussian_blur_kernel_mutex());
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
    if (hiprtcCreateProgram(&program, kGaussianBlurKernelSource, "gaussian_blur_u8.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
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
    if (hipModuleGetFunction(&function, module, "gaussian_blur_u8") != hipSuccess) {
        hipModuleUnload(module);
        return {StatusCode::runtime_error, "hipModuleGetFunction failed"};
    }

    kernel.module = module;
    kernel.function = function;
    return Status::success();
}

Status compile_threshold_kernel() noexcept
{
    auto& kernel = threshold_kernel();
    if (kernel.function != nullptr) {
        return Status::success();
    }

    std::lock_guard<std::mutex> lock(threshold_kernel_mutex());
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
    if (hiprtcCreateProgram(&program, kThresholdKernelSource, "threshold_gray8.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
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
    if (hipModuleGetFunction(&function, module, "threshold_gray8") != hipSuccess) {
        hipModuleUnload(module);
        return {StatusCode::runtime_error, "hipModuleGetFunction failed"};
    }

    kernel.module = module;
    kernel.function = function;
    return Status::success();
}

Status compile_resize_nearest_kernel() noexcept
{
    auto& kernel = resize_nearest_kernel();
    if (kernel.function != nullptr) {
        return Status::success();
    }

    std::lock_guard<std::mutex> lock(resize_nearest_kernel_mutex());
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
    if (hiprtcCreateProgram(&program, kResizeNearestKernelSource, "resize_nearest_u8.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
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
    if (hipModuleGetFunction(&function, module, "resize_nearest_u8") != hipSuccess) {
        hipModuleUnload(module);
        return {StatusCode::runtime_error, "hipModuleGetFunction failed"};
    }

    kernel.module = module;
    kernel.function = function;
    return Status::success();
}

Status compile_resize_bilinear_kernel() noexcept
{
    auto& kernel = resize_bilinear_kernel();
    if (kernel.function != nullptr) {
        return Status::success();
    }

    std::lock_guard<std::mutex> lock(resize_bilinear_kernel_mutex());
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
    if (hiprtcCreateProgram(&program, kResizeBilinearKernelSource, "resize_bilinear_u8.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
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
    if (hipModuleGetFunction(&function, module, "resize_bilinear_u8") != hipSuccess) {
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

    if (auto status = dst.allocate(color_conversion_shape_from(src_shape, code)); !status.ok()) {
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

        constexpr unsigned int block_x = 16;
        constexpr unsigned int block_y = 16;
        const auto grid_x = static_cast<unsigned int>((width + static_cast<int>(block_x) - 1) / static_cast<int>(block_x));
        const auto grid_y = static_cast<unsigned int>((height + static_cast<int>(block_y) - 1) / static_cast<int>(block_y));

        if (code == ColorConversion::bgr_to_gray || code == ColorConversion::rgb_to_gray) {
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

            if (hipModuleLaunchKernel(cvt_color_kernel().gray_function, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr) != hipSuccess) {
                dst.release();
                return {StatusCode::runtime_error, "hipModuleLaunchKernel failed"};
            }
        } else {
            void* args[] = {
                const_cast<std::uint8_t**>(&src_ptr),
                &dst_ptr,
                const_cast<int*>(&width),
                const_cast<int*>(&height),
                const_cast<unsigned long long*>(&src_step),
                const_cast<unsigned long long*>(&dst_step),
            };

            if (hipModuleLaunchKernel(cvt_color_kernel().swap_rb_function, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr) != hipSuccess) {
                dst.release();
                return {StatusCode::runtime_error, "hipModuleLaunchKernel failed"};
            }
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

Status resize(const GpuMat& src, GpuMat& dst, int width, int height, ResizeInterpolation interpolation) noexcept
{
    if (src.empty()) {
        return {StatusCode::invalid_argument, "source GpuMat is empty"};
    }

    if (width <= 0 || height <= 0) {
        return {StatusCode::invalid_argument, "destination size must be positive"};
    }

    if (interpolation != ResizeInterpolation::nearest && interpolation != ResizeInterpolation::bilinear) {
        return {StatusCode::invalid_argument, "unsupported resize interpolation"};
    }

    const auto src_shape = src.shape();
    if (!is_supported_resize_shape(src_shape)) {
        return {StatusCode::invalid_argument, "unsupported resize image format"};
    }

    if (auto status = dst.allocate(resized_shape_from(src_shape, width, height)); !status.ok()) {
        return status;
    }

#if HIPCV_HAS_HIP
    try {
        if (interpolation == ResizeInterpolation::nearest) {
            if (auto status = compile_resize_nearest_kernel(); !status.ok()) {
                dst.release();
                return status;
            }
        } else if (auto status = compile_resize_bilinear_kernel(); !status.ok()) {
            dst.release();
            return status;
        }

        const auto dst_shape = dst.shape();
        const auto* src_ptr = static_cast<const std::uint8_t*>(src.data());
        auto* dst_ptr = static_cast<std::uint8_t*>(dst.data());
        const int src_width = src_shape.width;
        const int src_height = src_shape.height;
        const int dst_width = dst_shape.width;
        const int dst_height = dst_shape.height;
        const int channels = src_shape.channels;
        const auto src_step = static_cast<unsigned long long>(src_shape.step_bytes);
        const auto dst_step = static_cast<unsigned long long>(dst_shape.step_bytes);

        void* args[] = {
            const_cast<std::uint8_t**>(&src_ptr),
            &dst_ptr,
            const_cast<int*>(&src_width),
            const_cast<int*>(&src_height),
            const_cast<int*>(&dst_width),
            const_cast<int*>(&dst_height),
            const_cast<int*>(&channels),
            const_cast<unsigned long long*>(&src_step),
            const_cast<unsigned long long*>(&dst_step),
        };

        constexpr unsigned int block_x = 16;
        constexpr unsigned int block_y = 16;
        const auto grid_x = static_cast<unsigned int>((dst_width + static_cast<int>(block_x) - 1) / static_cast<int>(block_x));
        const auto grid_y = static_cast<unsigned int>((dst_height + static_cast<int>(block_y) - 1) / static_cast<int>(block_y));

        const auto function = interpolation == ResizeInterpolation::nearest ? resize_nearest_kernel().function : resize_bilinear_kernel().function;
        if (hipModuleLaunchKernel(function, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr) != hipSuccess) {
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
        return {StatusCode::runtime_error, "resize failed"};
    }
#else
    return {StatusCode::hip_not_enabled, "hipcv was built without HIP support"};
#endif
}

Status threshold(const GpuMat& src, GpuMat& dst, std::uint8_t threshold_value, std::uint8_t max_value, ThresholdType type) noexcept
{
    if (src.empty()) {
        return {StatusCode::invalid_argument, "source GpuMat is empty"};
    }

    if (type != ThresholdType::binary && type != ThresholdType::binary_inv) {
        return {StatusCode::invalid_argument, "unsupported threshold type"};
    }

    const auto src_shape = src.shape();
    if (!is_supported_threshold_shape(src_shape)) {
        return {StatusCode::invalid_argument, "threshold currently supports gray8 only"};
    }

    if (auto status = dst.allocate(same_shape_as(src_shape)); !status.ok()) {
        return status;
    }

#if HIPCV_HAS_HIP
    try {
        if (auto status = compile_threshold_kernel(); !status.ok()) {
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
        const auto threshold_arg = static_cast<unsigned int>(threshold_value);
        const auto max_arg = static_cast<unsigned int>(max_value);
        const int inverse = type == ThresholdType::binary_inv ? 1 : 0;

        void* args[] = {
            const_cast<std::uint8_t**>(&src_ptr),
            &dst_ptr,
            const_cast<int*>(&width),
            const_cast<int*>(&height),
            const_cast<unsigned long long*>(&src_step),
            const_cast<unsigned long long*>(&dst_step),
            const_cast<unsigned int*>(&threshold_arg),
            const_cast<unsigned int*>(&max_arg),
            const_cast<int*>(&inverse),
        };

        constexpr unsigned int block_x = 16;
        constexpr unsigned int block_y = 16;
        const auto grid_x = static_cast<unsigned int>((width + static_cast<int>(block_x) - 1) / static_cast<int>(block_x));
        const auto grid_y = static_cast<unsigned int>((height + static_cast<int>(block_y) - 1) / static_cast<int>(block_y));

        if (hipModuleLaunchKernel(threshold_kernel().function, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr) != hipSuccess) {
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
        return {StatusCode::runtime_error, "threshold failed"};
    }
#else
    return {StatusCode::hip_not_enabled, "hipcv was built without HIP support"};
#endif
}

Status blur(const GpuMat& src, GpuMat& dst, int kernel_width, int kernel_height) noexcept
{
    if (src.empty()) {
        return {StatusCode::invalid_argument, "source GpuMat is empty"};
    }

    if (kernel_width <= 0 || kernel_height <= 0 || kernel_width % 2 == 0 || kernel_height % 2 == 0) {
        return {StatusCode::invalid_argument, "blur kernel size must be positive and odd"};
    }

    const auto src_shape = src.shape();
    if (!is_supported_blur_shape(src_shape)) {
        return {StatusCode::invalid_argument, "unsupported blur image format"};
    }

    if (auto status = dst.allocate(same_shape_as(src_shape)); !status.ok()) {
        return status;
    }

#if HIPCV_HAS_HIP
    try {
        if (auto status = compile_blur_kernel(); !status.ok()) {
            dst.release();
            return status;
        }

        const auto dst_shape = dst.shape();
        const auto* src_ptr = static_cast<const std::uint8_t*>(src.data());
        auto* dst_ptr = static_cast<std::uint8_t*>(dst.data());
        const int width = src_shape.width;
        const int height = src_shape.height;
        const int channels = src_shape.channels;
        const auto src_step = static_cast<unsigned long long>(src_shape.step_bytes);
        const auto dst_step = static_cast<unsigned long long>(dst_shape.step_bytes);

        void* args[] = {
            const_cast<std::uint8_t**>(&src_ptr),
            &dst_ptr,
            const_cast<int*>(&width),
            const_cast<int*>(&height),
            const_cast<unsigned long long*>(&src_step),
            const_cast<unsigned long long*>(&dst_step),
            const_cast<int*>(&channels),
            &kernel_width,
            &kernel_height,
        };

        constexpr unsigned int block_x = 16;
        constexpr unsigned int block_y = 16;
        const auto grid_x = static_cast<unsigned int>((width + static_cast<int>(block_x) - 1) / static_cast<int>(block_x));
        const auto grid_y = static_cast<unsigned int>((height + static_cast<int>(block_y) - 1) / static_cast<int>(block_y));

        if (hipModuleLaunchKernel(blur_kernel().function, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr) != hipSuccess) {
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
        return {StatusCode::runtime_error, "blur failed"};
    }
#else
    return {StatusCode::hip_not_enabled, "hipcv was built without HIP support"};
#endif
}

Status gaussianBlur(const GpuMat& src, GpuMat& dst, int kernel_width, int kernel_height) noexcept
{
    if (src.empty()) {
        return {StatusCode::invalid_argument, "source GpuMat is empty"};
    }

    if (kernel_width != kernel_height) {
        return {StatusCode::invalid_argument, "gaussianBlur currently requires a square kernel"};
    }

    if (kernel_width != 3 && kernel_width != 5) {
        return {StatusCode::invalid_argument, "gaussianBlur currently supports 3x3 and 5x5 kernels"};
    }

    const auto src_shape = src.shape();
    if (!is_supported_gaussian_blur_shape(src_shape)) {
        return {StatusCode::invalid_argument, "unsupported gaussianBlur image format"};
    }

    if (auto status = dst.allocate(same_shape_as(src_shape)); !status.ok()) {
        return status;
    }

#if HIPCV_HAS_HIP
    try {
        if (auto status = compile_gaussian_blur_kernel(); !status.ok()) {
            dst.release();
            return status;
        }

        const auto dst_shape = dst.shape();
        const auto* src_ptr = static_cast<const std::uint8_t*>(src.data());
        auto* dst_ptr = static_cast<std::uint8_t*>(dst.data());
        const int width = src_shape.width;
        const int height = src_shape.height;
        const int channels = src_shape.channels;
        const int kernel_size = kernel_width;
        const auto src_step = static_cast<unsigned long long>(src_shape.step_bytes);
        const auto dst_step = static_cast<unsigned long long>(dst_shape.step_bytes);

        void* args[] = {
            const_cast<std::uint8_t**>(&src_ptr),
            &dst_ptr,
            const_cast<int*>(&width),
            const_cast<int*>(&height),
            const_cast<unsigned long long*>(&src_step),
            const_cast<unsigned long long*>(&dst_step),
            const_cast<int*>(&channels),
            const_cast<int*>(&kernel_size),
        };

        constexpr unsigned int block_x = 16;
        constexpr unsigned int block_y = 16;
        const auto grid_x = static_cast<unsigned int>((width + static_cast<int>(block_x) - 1) / static_cast<int>(block_x));
        const auto grid_y = static_cast<unsigned int>((height + static_cast<int>(block_y) - 1) / static_cast<int>(block_y));

        if (hipModuleLaunchKernel(gaussian_blur_kernel().function, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr) != hipSuccess) {
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
        return {StatusCode::runtime_error, "gaussianBlur failed"};
    }
#else
    return {StatusCode::hip_not_enabled, "hipcv was built without HIP support"};
#endif
}

} // namespace hipcv
