# Roadmap

## Phase 0: Research and Design

- Define the target GPU and ROCm versions.
- Review the OpenCV CUDA API surface.
- Identify overlap with RPP, rocAL, and MIVisionX.
- Finalize the MVP function list.

## Phase 1: Core Infrastructure

- `GpuMat` data structure
- HIP memory allocation
- `upload` / `download`
- `Stream` and async execution model
- Error handling
- CMake build sistemi
- C++ smoke test

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

Example pipeline:

```text
upload -> resize -> cvtColor -> gaussianBlur -> download
```

## Phase 4: Python API

- pybind11 integration
- NumPy array input/output support
- Simple Python examples
- Packaging experiment

## Phase 5: OpenCV Compatibility

- `cv::Mat` and `hipcv::GpuMat` conversions
- Low-friction migration path for OpenCV-based projects
- Evaluate a `cv2.cuda`-like naming strategy
