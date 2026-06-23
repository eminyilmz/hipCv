#include "hipcv/hipcv.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
    if (!hipcv::built_with_hip()) {
        std::cout << "HIP backend is disabled; rebuild with HIPCV_ENABLE_HIP=ON.\n";
        return 0;
    }

    const auto devices = hipcv::query_devices();
    if (devices.device_count <= 0) {
        std::cout << "No HIP device is available.\n";
        return 0;
    }

    const std::vector<std::uint8_t> gray = {
        1, 2, 3,
        4, 5, 6,
    };

    const hipcv::ImageShape shape{
        3,
        2,
        1,
        0,
        hipcv::PixelFormat::gray8,
    };

    hipcv::GpuMat src;
    if (auto status = src.upload(gray.data(), shape); !status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return 1;
    }

    hipcv::GpuMat resized;
    if (auto status = hipcv::resize(src, resized, 6, 4); !status.ok()) {
        std::cerr << "resize failed: " << status.message() << '\n';
        return 1;
    }

    std::vector<std::uint8_t> output(6 * 4);
    if (auto status = resized.download(output.data(), output.size()); !status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "resize nearest 3x2 -> 6x4:\n";
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 6; ++x) {
            std::cout << static_cast<int>(output[static_cast<std::size_t>(y * 6 + x)]) << ' ';
        }
        std::cout << '\n';
    }

    hipcv::GpuMat bilinear;
    if (auto status = hipcv::resize(src, bilinear, 5, 3, hipcv::ResizeInterpolation::bilinear); !status.ok()) {
        std::cerr << "resize bilinear failed: " << status.message() << '\n';
        return 1;
    }

    std::vector<std::uint8_t> bilinear_output(5 * 3);
    if (auto status = bilinear.download(bilinear_output.data(), bilinear_output.size()); !status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "resize bilinear 3x2 -> 5x3:\n";
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 5; ++x) {
            std::cout << static_cast<int>(bilinear_output[static_cast<std::size_t>(y * 5 + x)]) << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}
