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

std::vector<std::uint8_t> threshold_reference(
    const std::vector<std::uint8_t>& src,
    std::uint8_t threshold_value,
    std::uint8_t max_value,
    hipcv::ThresholdType type)
{
    std::vector<std::uint8_t> dst;
    dst.reserve(src.size());

    for (const auto value : src) {
        const bool pass = value > threshold_value;
        const bool output_max = type == hipcv::ThresholdType::binary ? pass : !pass;
        dst.push_back(output_max ? max_value : 0);
    }

    return dst;
}

bool run_case(hipcv::ThresholdType type)
{
    const std::vector<std::uint8_t> input = {
        0, 50, 100, 101,
        128, 180, 220, 255,
    };

    const hipcv::ImageShape shape{
        4,
        2,
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
    status = hipcv::threshold(src, output_gpu, 100, 200, type);
    if (!status.ok()) {
        std::cerr << "threshold failed: " << status.message() << '\n';
        return false;
    }

    const auto output_shape = output_gpu.shape();
    expect(output_shape.width == shape.width, "threshold width should match source");
    expect(output_shape.height == shape.height, "threshold height should match source");
    expect(output_shape.channels == shape.channels, "threshold channels should match source");
    expect(output_shape.format == shape.format, "threshold format should match source");

    std::vector<std::uint8_t> output(input.size());
    status = output_gpu.download(output.data(), output.size());
    if (!status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return false;
    }

    const auto expected = threshold_reference(input, 100, 200, type);
    if (output != expected) {
        std::cerr << "threshold output mismatch\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!hipcv::built_with_hip()) {
        std::cout << "Skipping threshold GPU checks because HIP is disabled.\n";
        return 0;
    }

    const auto devices = hipcv::query_devices();
    if (devices.device_count <= 0) {
        std::cout << "Skipping threshold GPU checks because no HIP device is available.\n";
        return 0;
    }

    expect(run_case(hipcv::ThresholdType::binary), "binary threshold should match reference");
    expect(run_case(hipcv::ThresholdType::binary_inv), "binary_inv threshold should match reference");

    return failures == 0 ? 0 : 1;
}
