# OpenCV CUDA Migration Notes

`hipcv` is not a drop-in replacement for OpenCV CUDA. It is a Windows-first AMD
HIP image-processing layer that borrows familiar concepts where they make AMD
GPU workflows easier to read and adopt.

Use this document as a practical map for porting small preprocessing paths, not
as a compatibility promise.

## Concept Mapping

| OpenCV CUDA Concept | hipcv Concept | Current Status |
| --- | --- | --- |
| `cv::cuda::GpuMat` | `hipcv::GpuMat` | Move-only device buffer |
| `GpuMat::upload` | `GpuMat::upload` | Host-to-device copy |
| `GpuMat::download` | `GpuMat::download` | Device-to-host copy |
| `GpuMat::copyTo` | `GpuMat::copyTo` | Device-to-device copy |
| `cv::cuda::cvtColor` | `hipcv::cvtColor` | `BGR2GRAY`, `RGB2GRAY`, `BGR2RGB`, `RGB2BGR` |
| `cv::cuda::resize` | `hipcv::resize` | Nearest neighbor, bilinear |
| `cv::cuda::threshold` | `hipcv::threshold` | Binary, binary inverse for `gray8` |
| `cv::cuda::blur` | `hipcv::blur` | Box blur for `gray8` |
| `cv::cuda::GaussianBlur` | `hipcv::gaussianBlur` | 3x3 and 5x5 for `gray8` |
| `cv::cuda::Stream` | Not exposed yet | Planned carefully later |

## Basic Flow

An OpenCV CUDA preprocessing flow often looks like this:

```cpp
cv::cuda::GpuMat gpu_src;
cv::cuda::GpuMat gpu_small;
cv::cuda::GpuMat gpu_gray;
cv::cuda::GpuMat gpu_binary;

gpu_src.upload(host_bgr);
cv::cuda::resize(gpu_src, gpu_small, cv::Size(640, 360));
cv::cuda::cvtColor(gpu_small, gpu_gray, cv::COLOR_BGR2GRAY);
cv::cuda::threshold(gpu_gray, gpu_binary, 96, 255, cv::THRESH_BINARY);
gpu_binary.download(host_output);
```

The current `hipcv` equivalent is explicit about shape and status handling:

```cpp
#include "hipcv/hipcv.hpp"

hipcv::ImageShape bgr_shape{
    1920,
    1080,
    3,
    0,
    hipcv::PixelFormat::bgr8,
};

hipcv::GpuMat src;
hipcv::GpuMat small;
hipcv::GpuMat gray;
hipcv::GpuMat binary;

auto status = src.upload(host_bgr.data(), bgr_shape);
if (!status.ok()) {
    // handle status.message()
}

status = hipcv::resize(src, small, 640, 360);
status = hipcv::cvtColor(small, gray, hipcv::ColorConversion::bgr_to_gray);
status = hipcv::threshold(gray, binary, 96, 255);

std::vector<std::uint8_t> host_output(binary.size_bytes());
status = binary.download(host_output.data(), host_output.size());
```

The runnable example is:

```powershell
.\build\windows-vs2022\Release\hipcv_preprocess_pipeline.exe
```

## Important Differences

### Status Values Instead Of Exceptions

`hipcv` MVP functions return `hipcv::Status`. Check `status.ok()` and inspect
`status.code()` or `status.message()` on failure.

### Destination Allocation

Current operations allocate `dst` internally. This keeps early examples simple
and avoids requiring callers to duplicate output-shape logic.

### Format Scope

The MVP is intentionally narrow:

- `cvtColor` supports `BGR2GRAY`, `RGB2GRAY`, `BGR2RGB`, and `RGB2BGR`.
- `resize` supports nearest neighbor and bilinear interpolation.
- `threshold`, `blur`, and `gaussianBlur` currently operate on `gray8`.
- `gaussianBlur` supports 3x3 and 5x5 square kernels.

See [Supported operations](supported-operations.md) for the current matrix.

### Streams And Async Work

OpenCV CUDA code often passes `cv::cuda::Stream`. `hipcv` does not expose stream
control yet. The public API should keep HIP details contained until stream
ownership and Python binding needs are clearer.

### CPU `cv::Mat` Interop

`hipcv` does not require OpenCV types. Upload from any contiguous host buffer
that matches an `ImageShape`. For OpenCV users, that means passing `mat.data`
and a matching shape when the source matrix is continuous.

## Porting Checklist

1. Identify which parts of the OpenCV CUDA pipeline are already in the
   supported operation matrix.
2. Replace `cv::cuda::GpuMat` ownership with `hipcv::GpuMat`.
3. Convert `cv::Mat` metadata into `hipcv::ImageShape`.
4. Replace operation constants with `hipcv` enums.
5. Check every returned `Status`.
6. Keep intermediate images on the GPU until the final `download`.

## Current Best-Fit Use Case

`hipcv` is currently best suited for small AMD GPU preprocessing paths such as:

```text
BGR8 host image
-> upload
-> resize nearest or bilinear
-> BGR2GRAY
-> gaussianBlur 3x3
-> threshold binary
-> download
```

For unsupported OpenCV CUDA features, keep the fallback explicit for now rather
than hiding missing behavior behind partial compatibility.
