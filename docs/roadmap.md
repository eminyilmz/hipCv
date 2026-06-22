# Roadmap

## Phase 0: Research and Design

- Define the target Windows, GPU, and HIP SDK versions.
- Validate the Windows HIP SDK installation flow.
- Verify `hipInfo`, `hipconfig`, and a minimal HIP kernel on Windows.
- Review the OpenCV CUDA API surface.
- Identify which ROCm/RPP pieces are usable from HIP SDK for Windows.
- Finalize the MVP function list.

## Phase 1: Core Infrastructure

- `GpuMat` data structure
- HIP memory allocation on Windows
- `upload` / `download`
- `Stream` and async execution model
- Error handling
- CMake build system
- Visual Studio generator support
- C++ smoke test on Windows

## Phase 2: First Image Processing Functions

- `cvtColor`
- `resize`
- `threshold`
- `blur`
- `gaussianBlur`

Each function will include an accuracy test against OpenCV CPU output.

## Phase 3: Pipeline and Benchmarking

- Single-function benchmarks
- Chained pipeline benchmarks
- 720p, 1080p, and 4K tests
- Separate measurement for upload/download overhead
- Windows GPU timing validation

Example pipeline:

```text
upload -> resize -> cvtColor -> gaussianBlur -> download
```

## Phase 4: Python API

- pybind11 integration
- NumPy array input/output support
- Simple Python examples
- Windows wheel packaging experiment

## Phase 5: OpenCV Compatibility

- `cv::Mat` and `hipcv::GpuMat` conversions
- Low-friction migration path for OpenCV-based projects
- Evaluate a `cv2.cuda`-like naming strategy

## Phase 6: Linux Portability Check

- Verify the same core API on Linux ROCm.
- Keep platform-specific code isolated.
- Document differences between HIP SDK for Windows and full ROCm on Linux.
