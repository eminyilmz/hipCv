#include "hipcv/hipcv.hpp"

#include <cstdint>
#include <iostream>
#include <utility>
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

bool expect_ok(const hipcv::Status& status, const char* message)
{
    if (!status.ok()) {
        std::cerr << "FAIL: " << message << ": " << status.message() << '\n';
        ++failures;
        return false;
    }

    return true;
}

bool check_upload_download()
{
    const std::vector<std::uint8_t> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };

    const hipcv::ImageShape shape{
        4,
        3,
        1,
        0,
        hipcv::PixelFormat::gray8,
    };

    hipcv::GpuMat gpu;
    if (!expect_ok(gpu.upload(input.data(), shape), "upload should succeed")) {
        return false;
    }

    const auto gpu_shape = gpu.shape();
    expect(!gpu.empty(), "uploaded GpuMat should not be empty");
    expect(gpu.data() != nullptr, "uploaded GpuMat should have device data");
    expect(gpu_shape.width == shape.width, "uploaded width should match");
    expect(gpu_shape.height == shape.height, "uploaded height should match");
    expect(gpu_shape.channels == shape.channels, "uploaded channels should match");
    expect(gpu_shape.step_bytes == 4, "packed gray8 step should be normalized");
    expect(gpu.size_bytes() == input.size(), "uploaded byte size should match input");

    std::vector<std::uint8_t> output(input.size());
    if (!expect_ok(gpu.download(output.data(), output.size()), "download should succeed")) {
        return false;
    }

    expect(output == input, "downloaded bytes should match uploaded bytes");
    return output == input;
}

bool check_copy_to()
{
    const std::vector<std::uint8_t> input = {
        10, 20, 30, 40, 50, 60,
        70, 80, 90, 100, 110, 120,
    };

    const hipcv::ImageShape shape{
        2,
        2,
        3,
        0,
        hipcv::PixelFormat::bgr8,
    };

    hipcv::GpuMat src;
    if (!expect_ok(src.upload(input.data(), shape), "source upload should succeed")) {
        return false;
    }

    hipcv::GpuMat dst;
    if (!expect_ok(src.copyTo(dst), "copyTo should succeed")) {
        return false;
    }

    const auto dst_shape = dst.shape();
    expect(!dst.empty(), "copied GpuMat should not be empty");
    expect(dst.data() != nullptr, "copied GpuMat should have device data");
    expect(dst.data() != src.data(), "copyTo should allocate a distinct destination");
    expect(dst_shape.width == shape.width, "copied width should match");
    expect(dst_shape.height == shape.height, "copied height should match");
    expect(dst_shape.channels == shape.channels, "copied channels should match");
    expect(dst_shape.format == shape.format, "copied format should match");
    expect(dst.size_bytes() == input.size(), "copied byte size should match input");

    std::vector<std::uint8_t> output(input.size());
    if (!expect_ok(dst.download(output.data(), output.size()), "copied download should succeed")) {
        return false;
    }

    expect(output == input, "copied bytes should match uploaded bytes");
    return output == input;
}

bool check_move_semantics()
{
    const std::vector<std::uint8_t> input = {
        3, 1,
        4, 1,
    };

    const hipcv::ImageShape shape{
        2,
        2,
        1,
        0,
        hipcv::PixelFormat::gray8,
    };

    hipcv::GpuMat original;
    if (!expect_ok(original.upload(input.data(), shape), "move source upload should succeed")) {
        return false;
    }

    hipcv::GpuMat moved{std::move(original)};
    expect(original.empty(), "moved-from GpuMat should be empty");
    expect(!moved.empty(), "move-constructed GpuMat should own data");

    hipcv::GpuMat assigned;
    assigned = std::move(moved);
    expect(moved.empty(), "move-assigned source should be empty");
    expect(!assigned.empty(), "move-assigned destination should own data");

    std::vector<std::uint8_t> output(input.size());
    if (!expect_ok(assigned.download(output.data(), output.size()), "move-assigned download should succeed")) {
        return false;
    }

    expect(output == input, "move-assigned bytes should match uploaded bytes");
    return output == input;
}

} // namespace

int main()
{
    if (!hipcv::built_with_hip()) {
        std::cout << "Skipping GpuMat GPU roundtrip checks because HIP is disabled.\n";
        return 0;
    }

    const auto devices = hipcv::query_devices();
    if (devices.device_count <= 0) {
        std::cout << "Skipping GpuMat GPU roundtrip checks because no HIP device is available.\n";
        return 0;
    }

    expect(check_upload_download(), "GpuMat upload/download should roundtrip");
    expect(check_copy_to(), "GpuMat copyTo should preserve bytes");
    expect(check_move_semantics(), "GpuMat move semantics should preserve ownership");

    return failures == 0 ? 0 : 1;
}
