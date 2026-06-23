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

std::vector<std::uint8_t> resize_nearest_reference(
    const std::vector<std::uint8_t>& src,
    int src_width,
    int src_height,
    int channels,
    int dst_width,
    int dst_height)
{
    std::vector<std::uint8_t> dst(static_cast<std::size_t>(dst_width * dst_height * channels));

    for (int y = 0; y < dst_height; ++y) {
        const int src_y = (y * src_height) / dst_height;
        for (int x = 0; x < dst_width; ++x) {
            const int src_x = (x * src_width) / dst_width;

            for (int channel = 0; channel < channels; ++channel) {
                const auto src_index = static_cast<std::size_t>((src_y * src_width + src_x) * channels + channel);
                const auto dst_index = static_cast<std::size_t>((y * dst_width + x) * channels + channel);
                dst[dst_index] = src[src_index];
            }
        }
    }

    return dst;
}

std::vector<std::uint8_t> resize_bilinear_reference(
    const std::vector<std::uint8_t>& src,
    int src_width,
    int src_height,
    int channels,
    int dst_width,
    int dst_height)
{
    std::vector<std::uint8_t> dst(static_cast<std::size_t>(dst_width * dst_height * channels));

    const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
    const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);

    for (int y = 0; y < dst_height; ++y) {
        float src_fy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        if (src_fy < 0.0f) {
            src_fy = 0.0f;
        }

        const int y0 = static_cast<int>(src_fy);
        const int y1 = y0 + 1 >= src_height ? src_height - 1 : y0 + 1;
        const float ty = src_fy - static_cast<float>(y0);

        for (int x = 0; x < dst_width; ++x) {
            float src_fx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            if (src_fx < 0.0f) {
                src_fx = 0.0f;
            }

            const int x0 = static_cast<int>(src_fx);
            const int x1 = x0 + 1 >= src_width ? src_width - 1 : x0 + 1;
            const float tx = src_fx - static_cast<float>(x0);

            for (int channel = 0; channel < channels; ++channel) {
                const auto top_left = static_cast<float>(src[static_cast<std::size_t>((y0 * src_width + x0) * channels + channel)]);
                const auto top_right = static_cast<float>(src[static_cast<std::size_t>((y0 * src_width + x1) * channels + channel)]);
                const auto bottom_left = static_cast<float>(src[static_cast<std::size_t>((y1 * src_width + x0) * channels + channel)]);
                const auto bottom_right = static_cast<float>(src[static_cast<std::size_t>((y1 * src_width + x1) * channels + channel)]);
                const float top = top_left + ((top_right - top_left) * tx);
                const float bottom = bottom_left + ((bottom_right - bottom_left) * tx);
                const float value = top + ((bottom - top) * ty);

                const auto dst_index = static_cast<std::size_t>((y * dst_width + x) * channels + channel);
                dst[dst_index] = static_cast<std::uint8_t>(value + 0.5f);
            }
        }
    }

    return dst;
}

bool run_case(
    const std::vector<std::uint8_t>& input,
    hipcv::PixelFormat format,
    int channels,
    int dst_width,
    int dst_height,
    hipcv::ResizeInterpolation interpolation = hipcv::ResizeInterpolation::nearest)
{
    const hipcv::ImageShape shape{
        3,
        2,
        channels,
        0,
        format,
    };

    hipcv::GpuMat src;
    auto status = src.upload(input.data(), shape);
    if (!status.ok()) {
        std::cerr << "upload failed: " << status.message() << '\n';
        return false;
    }

    hipcv::GpuMat resized;
    status = hipcv::resize(src, resized, dst_width, dst_height, interpolation);
    if (!status.ok()) {
        std::cerr << "resize failed: " << status.message() << '\n';
        return false;
    }

    const auto resized_shape = resized.shape();
    expect(resized_shape.width == dst_width, "resized width should match requested width");
    expect(resized_shape.height == dst_height, "resized height should match requested height");
    expect(resized_shape.channels == channels, "resized channels should match source channels");
    expect(resized_shape.format == format, "resized format should match source format");

    std::vector<std::uint8_t> output(static_cast<std::size_t>(dst_width * dst_height * channels));
    status = resized.download(output.data(), output.size());
    if (!status.ok()) {
        std::cerr << "download failed: " << status.message() << '\n';
        return false;
    }

    const auto expected = interpolation == hipcv::ResizeInterpolation::nearest
        ? resize_nearest_reference(input, shape.width, shape.height, channels, dst_width, dst_height)
        : resize_bilinear_reference(input, shape.width, shape.height, channels, dst_width, dst_height);
    if (output != expected) {
        std::cerr << "resize output mismatch\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!hipcv::built_with_hip()) {
        std::cout << "Skipping resize GPU checks because HIP is disabled.\n";
        return 0;
    }

    const auto devices = hipcv::query_devices();
    if (devices.device_count <= 0) {
        std::cout << "Skipping resize GPU checks because no HIP device is available.\n";
        return 0;
    }

    const std::vector<std::uint8_t> gray = {
        1, 2, 3,
        4, 5, 6,
    };

    const std::vector<std::uint8_t> bgr = {
        10, 20, 30,
        40, 50, 60,
        70, 80, 90,
        11, 21, 31,
        41, 51, 61,
        71, 81, 91,
    };

    expect(run_case(gray, hipcv::PixelFormat::gray8, 1, 6, 4), "gray nearest resize should match reference");
    expect(run_case(bgr, hipcv::PixelFormat::bgr8, 3, 2, 1), "BGR nearest resize should match reference");
    expect(
        run_case(gray, hipcv::PixelFormat::gray8, 1, 5, 3, hipcv::ResizeInterpolation::bilinear),
        "gray bilinear resize should match reference");
    expect(
        run_case(bgr, hipcv::PixelFormat::bgr8, 3, 5, 3, hipcv::ResizeInterpolation::bilinear),
        "BGR bilinear resize should match reference");

    return failures == 0 ? 0 : 1;
}
