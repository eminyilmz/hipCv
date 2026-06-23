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
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
    };

    const hipcv::ImageShape shape{
        4,
        3,
        1,
        0,
        hipcv::PixelFormat::gray8,
    };

    hipcv::GpuMat src;
    if (auto status = src.upload(gray.data(), shape); !status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return 1;
    }

    hipcv::GpuMat blurred;
    if (auto status = hipcv::blur(src, blurred, 3, 3); !status.ok()) {
        std::cerr << "blur failed: " << status.message() << '\n';
        return 1;
    }

    std::vector<std::uint8_t> output(gray.size());
    if (auto status = blurred.download(output.data(), output.size()); !status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "blur 3x3:\n";
    for (int y = 0; y < shape.height; ++y) {
        for (int x = 0; x < shape.width; ++x) {
            std::cout << static_cast<int>(output[static_cast<std::size_t>(y * shape.width + x)]) << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}
