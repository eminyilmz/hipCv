#include "hipcv/hipcv.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect_code(const hipcv::Status& status, hipcv::StatusCode code, const char* message)
{
    if (status.code() != code) {
        std::cerr << "FAIL: " << message << " (got " << static_cast<int>(status.code()) << ")\n";
        ++failures;
    }
}

bool has_hip_device()
{
    if (!hipcv::built_with_hip()) {
        return false;
    }

    return hipcv::query_devices().device_count > 0;
}

hipcv::GpuMat upload_or_fail(const std::vector<std::uint8_t>& pixels, hipcv::ImageShape shape, const char* label)
{
    hipcv::GpuMat gpu;
    const auto status = gpu.upload(pixels.data(), shape);
    if (!status.ok()) {
        std::cerr << label << " upload failed: " << status.message() << '\n';
        ++failures;
    }
    return gpu;
}

void check_empty_sources()
{
    hipcv::GpuMat empty;
    hipcv::GpuMat dst;

    expect_code(
        hipcv::cvtColor(empty, dst, hipcv::ColorConversion::bgr_to_gray),
        hipcv::StatusCode::invalid_argument,
        "cvtColor should reject empty source");

    expect_code(
        hipcv::resize(empty, dst, 2, 2),
        hipcv::StatusCode::invalid_argument,
        "resize should reject empty source");

    expect_code(
        hipcv::threshold(empty, dst, 100, 255),
        hipcv::StatusCode::invalid_argument,
        "threshold should reject empty source");

    expect_code(
        hipcv::blur(empty, dst, 3, 3),
        hipcv::StatusCode::invalid_argument,
        "blur should reject empty source");

    expect_code(
        hipcv::gaussianBlur(empty, dst, 3, 3),
        hipcv::StatusCode::invalid_argument,
        "gaussianBlur should reject empty source");
}

void check_hip_validation_paths()
{
    const std::vector<std::uint8_t> gray_pixels = {
        10, 20,
        30, 40,
    };
    const hipcv::ImageShape gray_shape{
        2,
        2,
        1,
        0,
        hipcv::PixelFormat::gray8,
    };

    const std::vector<std::uint8_t> bgr_pixels = {
        10, 20, 30, 40, 50, 60,
        70, 80, 90, 100, 110, 120,
    };
    const hipcv::ImageShape bgr_shape{
        2,
        2,
        3,
        0,
        hipcv::PixelFormat::bgr8,
    };
    const hipcv::ImageShape unknown_shape{
        2,
        2,
        1,
        0,
        hipcv::PixelFormat::unknown,
    };

    auto gray = upload_or_fail(gray_pixels, gray_shape, "gray");
    auto bgr = upload_or_fail(bgr_pixels, bgr_shape, "bgr");
    auto unknown = upload_or_fail(gray_pixels, unknown_shape, "unknown");
    if (failures != 0) {
        return;
    }

    hipcv::GpuMat dst;

    expect_code(
        hipcv::cvtColor(gray, dst, hipcv::ColorConversion::bgr_to_gray),
        hipcv::StatusCode::invalid_argument,
        "cvtColor should reject gray source for BGR2GRAY");

    expect_code(
        hipcv::resize(gray, dst, 0, 2),
        hipcv::StatusCode::invalid_argument,
        "resize should reject non-positive width");

    expect_code(
        hipcv::resize(gray, dst, 2, 2, static_cast<hipcv::ResizeInterpolation>(999)),
        hipcv::StatusCode::invalid_argument,
        "resize should reject unsupported interpolation");

    expect_code(
        hipcv::threshold(bgr, dst, 100, 255),
        hipcv::StatusCode::invalid_argument,
        "threshold should reject non-gray source");

    expect_code(
        hipcv::threshold(gray, dst, 100, 255, static_cast<hipcv::ThresholdType>(999)),
        hipcv::StatusCode::invalid_argument,
        "threshold should reject unsupported type");

    expect_code(
        hipcv::blur(gray, dst, 2, 3),
        hipcv::StatusCode::invalid_argument,
        "blur should reject even kernel width");

    expect_code(
        hipcv::blur(unknown, dst, 3, 3),
        hipcv::StatusCode::invalid_argument,
        "blur should reject unsupported image format");

    expect_code(
        hipcv::gaussianBlur(gray, dst, 3, 5),
        hipcv::StatusCode::invalid_argument,
        "gaussianBlur should reject non-square kernels");

    expect_code(
        hipcv::gaussianBlur(gray, dst, 7, 7),
        hipcv::StatusCode::invalid_argument,
        "gaussianBlur should reject unsupported kernel sizes");

    expect_code(
        hipcv::gaussianBlur(bgr, dst, 3, 3),
        hipcv::StatusCode::invalid_argument,
        "gaussianBlur should reject non-gray source");
}

} // namespace

int main()
{
    check_empty_sources();

    if (!has_hip_device()) {
        std::cout << "Skipping format-specific imgproc validation checks because HIP is disabled or no device is available.\n";
        return failures == 0 ? 0 : 1;
    }

    check_hip_validation_paths();
    return failures == 0 ? 0 : 1;
}
