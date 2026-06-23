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

std::uint8_t gray_from_rgb(std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    return static_cast<std::uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
}

bool run_case(const std::vector<std::uint8_t>& input, hipcv::PixelFormat format, hipcv::ColorConversion conversion)
{
    const hipcv::ImageShape shape{
        3,
        2,
        3,
        0,
        format,
    };

    hipcv::GpuMat src;
    auto status = src.upload(input.data(), shape);
    if (!status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return false;
    }

    hipcv::GpuMat gray;
    status = hipcv::cvtColor(src, gray, conversion);
    if (!status.ok()) {
        std::cerr << "cvtColor failed: " << status.message() << '\n';
        return false;
    }

    const auto gray_shape = gray.shape();
    expect(gray_shape.width == shape.width, "gray width should match source");
    expect(gray_shape.height == shape.height, "gray height should match source");
    expect(gray_shape.channels == 1, "gray image should have one channel");
    expect(gray_shape.format == hipcv::PixelFormat::gray8, "gray image should use gray8 format");

    std::vector<std::uint8_t> output(static_cast<std::size_t>(shape.width * shape.height));
    status = gray.download(output.data(), output.size());
    if (!status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return false;
    }

    std::vector<std::uint8_t> expected;
    expected.reserve(output.size());

    for (std::size_t i = 0; i < input.size(); i += 3) {
        if (format == hipcv::PixelFormat::bgr8) {
            expected.push_back(gray_from_rgb(input[i + 2], input[i + 1], input[i]));
        } else {
            expected.push_back(gray_from_rgb(input[i], input[i + 1], input[i + 2]));
        }
    }

    if (output != expected) {
        std::cerr << "grayscale output mismatch\n";
        return false;
    }

    return true;
}

bool run_swap_case(
    const std::vector<std::uint8_t>& input,
    hipcv::PixelFormat input_format,
    hipcv::PixelFormat output_format,
    hipcv::ColorConversion conversion)
{
    const hipcv::ImageShape shape{
        3,
        2,
        3,
        0,
        input_format,
    };

    hipcv::GpuMat src;
    auto status = src.upload(input.data(), shape);
    if (!status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return false;
    }

    hipcv::GpuMat converted;
    status = hipcv::cvtColor(src, converted, conversion);
    if (!status.ok()) {
        std::cerr << "cvtColor swap failed: " << status.message() << '\n';
        return false;
    }

    const auto converted_shape = converted.shape();
    expect(converted_shape.width == shape.width, "converted width should match source");
    expect(converted_shape.height == shape.height, "converted height should match source");
    expect(converted_shape.channels == 3, "converted image should have three channels");
    expect(converted_shape.format == output_format, "converted image should use requested color format");

    std::vector<std::uint8_t> output(input.size());
    status = converted.download(output.data(), output.size());
    if (!status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return false;
    }

    std::vector<std::uint8_t> expected(input.size());
    for (std::size_t i = 0; i < input.size(); i += 3) {
        expected[i + 0] = input[i + 2];
        expected[i + 1] = input[i + 1];
        expected[i + 2] = input[i + 0];
    }

    if (output != expected) {
        std::cerr << "channel swap output mismatch\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!hipcv::built_with_hip()) {
        std::cout << "Skipping cvtColor GPU checks because HIP is disabled.\n";
        return 0;
    }

    const auto devices = hipcv::query_devices();
    if (devices.device_count <= 0) {
        std::cout << "Skipping cvtColor GPU checks because no HIP device is available.\n";
        return 0;
    }

    const std::vector<std::uint8_t> bgr = {
        0, 0, 255,
        0, 255, 0,
        255, 0, 0,
        10, 20, 30,
        200, 100, 50,
        7, 8, 9,
    };

    const std::vector<std::uint8_t> rgb = {
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        30, 20, 10,
        50, 100, 200,
        9, 8, 7,
    };

    expect(run_case(bgr, hipcv::PixelFormat::bgr8, hipcv::ColorConversion::bgr_to_gray), "BGR to gray should match reference");
    expect(run_case(rgb, hipcv::PixelFormat::rgb8, hipcv::ColorConversion::rgb_to_gray), "RGB to gray should match reference");
    expect(
        run_swap_case(bgr, hipcv::PixelFormat::bgr8, hipcv::PixelFormat::rgb8, hipcv::ColorConversion::bgr_to_rgb),
        "BGR to RGB should swap red and blue channels");
    expect(
        run_swap_case(rgb, hipcv::PixelFormat::rgb8, hipcv::PixelFormat::bgr8, hipcv::ColorConversion::rgb_to_bgr),
        "RGB to BGR should swap red and blue channels");

    return failures == 0 ? 0 : 1;
}
