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

int gaussian_weight(int offset, int kernel_size)
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

std::vector<std::uint8_t> gaussian_blur_reference(
    const std::vector<std::uint8_t>& src,
    int width,
    int height,
    int kernel_size)
{
    std::vector<std::uint8_t> dst(src.size());
    const int radius = kernel_size / 2;
    const unsigned int weight_sum = kernel_size == 3 ? 16u : 256u;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned int sum = 0;

            for (int ky = -radius; ky <= radius; ++ky) {
                const int src_y = clamp(y + ky, 0, height - 1);
                const auto weight_y = static_cast<unsigned int>(gaussian_weight(ky, kernel_size));

                for (int kx = -radius; kx <= radius; ++kx) {
                    const int src_x = clamp(x + kx, 0, width - 1);
                    const auto weight_x = static_cast<unsigned int>(gaussian_weight(kx, kernel_size));
                    const auto weight = weight_x * weight_y;
                    sum += weight * static_cast<unsigned int>(src[static_cast<std::size_t>(src_y * width + src_x)]);
                }
            }

            dst[static_cast<std::size_t>(y * width + x)] = static_cast<std::uint8_t>((sum + (weight_sum / 2u)) / weight_sum);
        }
    }

    return dst;
}

bool run_case(int kernel_size)
{
    const std::vector<std::uint8_t> input = {
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,
        110, 120, 130, 140, 150,
        160, 170, 180, 190, 200,
    };

    const hipcv::ImageShape shape{
        5,
        4,
        1,
        0,
        hipcv::PixelFormat::gray8,
    };

    hipcv::GpuMat src;
    auto status = src.upload(input.data(), shape);
    if (!status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return false;
    }

    hipcv::GpuMat output_gpu;
    status = hipcv::gaussianBlur(src, output_gpu, kernel_size, kernel_size);
    if (!status.ok()) {
        std::cerr << "gaussianBlur failed: " << status.message() << '\n';
        return false;
    }

    const auto output_shape = output_gpu.shape();
    expect(output_shape.width == shape.width, "gaussianBlur width should match source");
    expect(output_shape.height == shape.height, "gaussianBlur height should match source");
    expect(output_shape.channels == shape.channels, "gaussianBlur channels should match source");
    expect(output_shape.format == shape.format, "gaussianBlur format should match source");

    std::vector<std::uint8_t> output(input.size());
    status = output_gpu.download(output.data(), output.size());
    if (!status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return false;
    }

    const auto expected = gaussian_blur_reference(input, shape.width, shape.height, kernel_size);
    if (output != expected) {
        std::cerr << "gaussianBlur output mismatch for " << kernel_size << "x" << kernel_size << '\n';
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!hipcv::built_with_hip()) {
        std::cout << "Skipping gaussianBlur GPU checks because HIP is disabled.\n";
        return 0;
    }

    const auto devices = hipcv::query_devices();
    if (devices.device_count <= 0) {
        std::cout << "Skipping gaussianBlur GPU checks because no HIP device is available.\n";
        return 0;
    }

    expect(run_case(3), "3x3 gaussianBlur should match reference");
    expect(run_case(5), "5x5 gaussianBlur should match reference");

    return failures == 0 ? 0 : 1;
}
