#include "hipcv/hipcv.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::uint8_t> make_bgr_frame(int width, int height)
{
    std::vector<std::uint8_t> frame(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto index = static_cast<std::size_t>((y * width + x) * 3);
            frame[index + 0] = static_cast<std::uint8_t>((x + y) & 0xFF);
            frame[index + 1] = static_cast<std::uint8_t>((2 * x + y) & 0xFF);
            frame[index + 2] = static_cast<std::uint8_t>((x + 2 * y) & 0xFF);
        }
    }

    return frame;
}

void print_average(const std::string& label, double total_ms, int iterations)
{
    std::cout << std::left << std::setw(28) << label
              << std::right << std::fixed << std::setprecision(3)
              << (total_ms / static_cast<double>(iterations)) << " ms\n";
}

bool run_pipeline(const hipcv::GpuMat& src, hipcv::GpuMat& binary)
{
    hipcv::GpuMat resized;
    if (auto status = hipcv::resize(src, resized, 640, 360); !status.ok()) {
        std::cerr << "resize failed: " << status.message() << '\n';
        return false;
    }

    hipcv::GpuMat gray;
    if (auto status = hipcv::cvtColor(resized, gray, hipcv::ColorConversion::bgr_to_gray); !status.ok()) {
        std::cerr << "cvtColor failed: " << status.message() << '\n';
        return false;
    }

    hipcv::GpuMat smoothed;
    if (auto status = hipcv::gaussianBlur(gray, smoothed, 3, 3); !status.ok()) {
        std::cerr << "gaussianBlur failed: " << status.message() << '\n';
        return false;
    }

    if (auto status = hipcv::threshold(smoothed, binary, 96, 255); !status.ok()) {
        std::cerr << "threshold failed: " << status.message() << '\n';
        return false;
    }

    return true;
}

} // namespace

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

    constexpr int input_width = 1920;
    constexpr int input_height = 1080;
    constexpr int iterations = 20;

    const auto frame = make_bgr_frame(input_width, input_height);
    const hipcv::ImageShape input_shape{
        input_width,
        input_height,
        3,
        0,
        hipcv::PixelFormat::bgr8,
    };

    hipcv::GpuMat src;
    if (auto status = src.upload(frame.data(), input_shape); !status.ok()) {
        std::cerr << "initial upload failed: " << status.message() << '\n';
        return 1;
    }

    hipcv::GpuMat binary;
    if (!run_pipeline(src, binary)) {
        return 1;
    }

    std::vector<std::uint8_t> host_output(binary.size_bytes());
    if (auto status = binary.download(host_output.data(), host_output.size()); !status.ok()) {
        std::cerr << "initial download failed: " << status.message() << '\n';
        return 1;
    }

    double upload_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        hipcv::GpuMat upload_dst;
        const auto start = Clock::now();
        if (auto status = upload_dst.upload(frame.data(), input_shape); !status.ok()) {
            std::cerr << "upload failed: " << status.message() << '\n';
            return 1;
        }
        const auto end = Clock::now();
        upload_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double pipeline_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        hipcv::GpuMat pipeline_dst;
        const auto start = Clock::now();
        if (!run_pipeline(src, pipeline_dst)) {
            return 1;
        }
        const auto end = Clock::now();
        pipeline_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double download_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        if (auto status = binary.download(host_output.data(), host_output.size()); !status.ok()) {
            std::cerr << "download failed: " << status.message() << '\n';
            return 1;
        }
        const auto end = Clock::now();
        download_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::cout << "hipcv preprocess benchmark\n";
    std::cout << "input: " << input_width << "x" << input_height << " BGR8\n";
    std::cout << "pipeline: resize 640x360 -> BGR2GRAY -> gaussianBlur 3x3 -> threshold\n";
    std::cout << "iterations: " << iterations << "\n\n";
    print_average("upload H2D", upload_ms, iterations);
    print_average("GPU pipeline", pipeline_ms, iterations);
    print_average("download D2H", download_ms, iterations);

    return 0;
}
