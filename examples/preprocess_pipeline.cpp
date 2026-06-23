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

    const std::vector<std::uint8_t> bgr = {
        255, 0, 0,     180, 40, 10,   80, 120, 40,   20, 200, 20,
        240, 30, 30,   160, 80, 30,   90, 150, 60,   30, 220, 80,
        200, 70, 70,   120, 120, 90,  70, 180, 120,  40, 240, 160,
    };

    const hipcv::ImageShape bgr_shape{
        4,
        3,
        3,
        0,
        hipcv::PixelFormat::bgr8,
    };

    hipcv::GpuMat src;
    if (auto status = src.upload(bgr.data(), bgr_shape); !status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return 1;
    }

    hipcv::GpuMat resized;
    if (auto status = hipcv::resize(src, resized, 8, 6); !status.ok()) {
        std::cerr << "resize failed: " << status.message() << '\n';
        return 1;
    }

    hipcv::GpuMat gray;
    if (auto status = hipcv::cvtColor(resized, gray, hipcv::ColorConversion::bgr_to_gray); !status.ok()) {
        std::cerr << "cvtColor failed: " << status.message() << '\n';
        return 1;
    }

    hipcv::GpuMat smoothed;
    if (auto status = hipcv::gaussianBlur(gray, smoothed, 3, 3); !status.ok()) {
        std::cerr << "gaussianBlur failed: " << status.message() << '\n';
        return 1;
    }

    hipcv::GpuMat binary;
    if (auto status = hipcv::threshold(smoothed, binary, 96, 255); !status.ok()) {
        std::cerr << "threshold failed: " << status.message() << '\n';
        return 1;
    }

    const auto output_shape = binary.shape();
    std::vector<std::uint8_t> output(binary.size_bytes());
    if (auto status = binary.download(output.data(), output.size()); !status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "preprocess pipeline: upload -> resize -> cvtColor -> gaussianBlur -> threshold -> download\n";
    for (int y = 0; y < output_shape.height; ++y) {
        for (int x = 0; x < output_shape.width; ++x) {
            std::cout << static_cast<int>(output[static_cast<std::size_t>(y * output_shape.width + x)]) << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}
