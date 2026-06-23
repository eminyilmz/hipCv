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
        0, 0, 255,
        0, 255, 0,
        255, 0, 0,
    };

    const hipcv::ImageShape shape{
        3,
        1,
        3,
        0,
        hipcv::PixelFormat::bgr8,
    };

    hipcv::GpuMat src;
    if (auto status = src.upload(bgr.data(), shape); !status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return 1;
    }

    hipcv::GpuMat gray;
    if (auto status = hipcv::cvtColor(src, gray, hipcv::ColorConversion::bgr_to_gray); !status.ok()) {
        std::cerr << "cvtColor failed: " << status.message() << '\n';
        return 1;
    }

    std::vector<std::uint8_t> output(3);
    if (auto status = gray.download(output.data(), output.size()); !status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "BGR2GRAY:";
    for (const auto value : output) {
        std::cout << ' ' << static_cast<int>(value);
    }
    std::cout << '\n';

    hipcv::GpuMat rgb;
    if (auto status = hipcv::cvtColor(src, rgb, hipcv::ColorConversion::bgr_to_rgb); !status.ok()) {
        std::cerr << "cvtColor swap failed: " << status.message() << '\n';
        return 1;
    }

    std::vector<std::uint8_t> rgb_output(bgr.size());
    if (auto status = rgb.download(rgb_output.data(), rgb_output.size()); !status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "BGR2RGB:";
    for (std::size_t i = 0; i < rgb_output.size(); i += 3) {
        std::cout << " (" << static_cast<int>(rgb_output[i])
                  << ',' << static_cast<int>(rgb_output[i + 1])
                  << ',' << static_cast<int>(rgb_output[i + 2]) << ')';
    }
    std::cout << '\n';

    return 0;
}
