#include "hipcv/hipcv.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

int clamp(int value, int low, int high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

std::vector<std::uint8_t> blur_reference(
    const std::vector<std::uint8_t>& src,
    int width,
    int height,
    int channels,
    int kernel_width,
    int kernel_height)
{
    std::vector<std::uint8_t> dst(src.size());
    const int radius_x = kernel_width / 2;
    const int radius_y = kernel_height / 2;
    const auto area = static_cast<unsigned int>(kernel_width * kernel_height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < channels; ++channel) {
                unsigned int sum = 0;

                for (int ky = 0; ky < kernel_height; ++ky) {
                    const int src_y = clamp(y + ky - radius_y, 0, height - 1);
                    for (int kx = 0; kx < kernel_width; ++kx) {
                        const int src_x = clamp(x + kx - radius_x, 0, width - 1);
                        sum += src[static_cast<std::size_t>((src_y * width + src_x) * channels + channel)];
                    }
                }

                dst[static_cast<std::size_t>((y * width + x) * channels + channel)] = static_cast<std::uint8_t>((sum + (area / 2u)) / area);
            }
        }
    }

    return dst;
}

bool run_case(
    const std::vector<std::uint8_t>& input,
    hipcv::PixelFormat format,
    int width,
    int height,
    int channels,
    int kernel_width,
    int kernel_height)
{
    const hipcv::ImageShape shape{
        width,
        height,
        channels,
        0,
        format,
    };

    hipcv::GpuMat src;
    auto status = src.upload(input.data(), shape);
    if (!status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return false;
    }

    hipcv::GpuMat output_gpu;
    status = hipcv::blur(src, output_gpu, kernel_width, kernel_height);
    if (!status.ok()) {
        std::cerr << "blur failed: " << status.message() << '\n';
        return false;
    }

    const auto output_shape = output_gpu.shape();
    expect(output_shape.width == shape.width, "blur width should match source");
    expect(output_shape.height == shape.height, "blur height should match source");
    expect(output_shape.channels == shape.channels, "blur channels should match source");
    expect(output_shape.format == shape.format, "blur format should match source");

    std::vector<std::uint8_t> output(input.size());
    status = output_gpu.download(output.data(), output.size());
    if (!status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return false;
    }

    const auto expected = blur_reference(input, shape.width, shape.height, shape.channels, kernel_width, kernel_height);
    if (output != expected) {
        std::cerr << "blur output mismatch\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!hipcv::built_with_hip()) {
        std::cout << "Skipping blur GPU checks because HIP is disabled.\n";
        return 0;
    }

    const auto devices = hipcv::query_devices();
    if (devices.device_count <= 0) {
        std::cout << "Skipping blur GPU checks because no HIP device is available.\n";
        return 0;
    }

    const std::vector<std::uint8_t> gray = {
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
    };

    const std::vector<std::uint8_t> bgr = {
        10, 20, 30, 40, 50, 60, 70, 80, 90,
        11, 21, 31, 41, 51, 61, 71, 81, 91,
        12, 22, 32, 42, 52, 62, 72, 82, 92,
    };

    expect(run_case(gray, hipcv::PixelFormat::gray8, 4, 3, 1, 3, 3), "3x3 gray blur should match reference");
    expect(run_case(gray, hipcv::PixelFormat::gray8, 4, 3, 1, 1, 3), "1x3 gray blur should match reference");
    expect(run_case(bgr, hipcv::PixelFormat::bgr8, 3, 3, 3, 3, 3), "3x3 BGR blur should match reference");

    return failures == 0 ? 0 : 1;
}
