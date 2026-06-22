#include "hipcv/gpu_mat.hpp"
#include "hipcv/version.hpp"

#include <cstdint>
#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_code(const hipcv::Status& status, hipcv::StatusCode code, const char* message)
{
    if (status.code() != code) {
        std::cerr << "FAIL: " << message << " (got " << static_cast<int>(status.code()) << ")\n";
        ++failures;
    }
}

} // namespace

int main()
{
    if (hipcv::built_with_hip()) {
        std::cout << "Skipping no-HIP GpuMat checks because HIP is enabled.\n";
        return 0;
    }

    hipcv::GpuMat src;
    hipcv::GpuMat dst;

    expect(src.empty(), "default GpuMat should be empty");
    expect(src.data() == nullptr, "default GpuMat data should be null");
    expect(src.size_bytes() == 0, "default GpuMat should have zero size");

    const hipcv::ImageShape valid_shape{
        2,
        2,
        3,
        0,
        hipcv::PixelFormat::bgr8,
    };

    auto status = src.allocate(valid_shape);
    expect_code(status, hipcv::StatusCode::hip_not_enabled, "valid allocation should fail clearly without HIP");
    expect(src.empty(), "failed allocation should leave GpuMat empty");

    status = src.allocate({0, 2, 3, 0, hipcv::PixelFormat::bgr8});
    expect_code(status, hipcv::StatusCode::invalid_argument, "zero width should be invalid");

    status = src.allocate({2, 2, 3, 2, hipcv::PixelFormat::bgr8});
    expect_code(status, hipcv::StatusCode::invalid_argument, "step smaller than packed row should be invalid");

    const std::uint8_t pixels[12] = {};
    status = src.upload(nullptr, valid_shape);
    expect_code(status, hipcv::StatusCode::invalid_argument, "null upload pointer should be invalid");

    status = src.upload(pixels, valid_shape);
    expect_code(status, hipcv::StatusCode::hip_not_enabled, "upload should fail clearly without HIP");
    expect(src.empty(), "failed upload should leave GpuMat empty");

    std::uint8_t output[12] = {};
    status = src.download(output, sizeof(output));
    expect_code(status, hipcv::StatusCode::invalid_argument, "download from empty GpuMat should be invalid");

    status = src.copyTo(dst);
    expect_code(status, hipcv::StatusCode::invalid_argument, "copy from empty GpuMat should be invalid");
    expect(dst.empty(), "failed copy should leave destination empty");

    return failures == 0 ? 0 : 1;
}
