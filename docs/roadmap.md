# Roadmap

`hipcv` is a Windows-first HIP image processing library for AMD GPUs. The
roadmap is organized around one rule: prove a small, reliable GPU-resident
workflow before expanding the API surface.

The first meaningful product milestone is:

```text
host image -> upload -> BGR/RGB to GRAY -> resize -> download
```

This proves memory ownership, HIP execution, image layout rules, correctness
tests, and the OpenCV-like developer experience.

## Phase 0: Platform Validation

Goal: prove that the Windows development environment can build and run the
project before adding more image operations.

### Tasks

- Install or locate CMake and Visual Studio 2022 Build Tools.
- Validate the no-HIP build preset.
- Install or locate AMD HIP SDK for Windows.
- Validate `hipInfo` and `hipconfig`.
- Validate the HIP build preset.
- Run the current smoke test in both no-HIP and HIP modes.
- Document exact tested versions:
  - Windows version
  - GPU model
  - AMD driver version
  - HIP SDK version
  - CMake version
  - Visual Studio version

### Deliverables

- Confirmed local no-HIP build.
- Confirmed local HIP build, if hardware and SDK are available.
- Updated Windows setup notes if the current instructions are incomplete.

### Exit Criteria

- `cmake --preset windows-vs2022-no-hip` succeeds.
- `cmake --build --preset windows-vs2022-no-hip-release` succeeds.
- `ctest --preset windows-vs2022-no-hip-release` succeeds.
- If HIP SDK is available, the same configure/build/test flow succeeds with
  `windows-vs2022`.

## Phase 1: Core Memory API

Goal: make `GpuMat` safe, predictable, and useful enough to support real image
operations.

### Tasks

- Finalize `ImageShape` validation rules.
- Decide supported MVP pixel formats:
  - `gray8`
  - `rgb8`
  - `bgr8`
  - optionally `rgba8` and `bgra8`
- Keep `GpuMat` move-only and RAII-owned.
- Implement or verify:
  - `allocate`
  - `upload`
  - `download`
  - `copyTo`
  - `release`
- Add convenience free functions:
  - `upload(...)`
  - `download(...)`
  - `copy(...)`
- Define how row stride works:
  - packed rows by default
  - explicit `step_bytes` support
  - validation for invalid stride values
- Add basic no-HIP behavior tests.

### Deliverables

- Stable memory API in `include/hipcv/gpu_mat.hpp`.
- Optional `include/hipcv/memory.hpp` for free-function helpers.
- Tests for invalid shapes, allocation behavior, and no-HIP failure behavior.

### Exit Criteria

- Empty matrices are handled safely.
- Invalid shapes return `StatusCode::invalid_argument`.
- No-HIP builds return `StatusCode::hip_not_enabled` for GPU operations.
- HIP builds can upload, copy, and download a small test image.

## Phase 2: First HIP Kernel

Goal: prove that the project can compile and run an actual image-processing
kernel on Windows HIP SDK.

### Recommended First Operation

Start with grayscale conversion:

```cpp
Status cvtColor(const GpuMat& src, GpuMat& dst, ColorConversion code) noexcept;
```

Initial conversion codes:

```cpp
ColorConversion::bgr_to_gray
ColorConversion::rgb_to_gray
```

### Tasks

- Add `include/hipcv/imgproc.hpp`.
- Add a small `ColorConversion` enum.
- Decide the HIP kernel build strategy:
  - CMake HIP language, or
  - HIP SDK compiler integration, or
  - `.cpp` source using HIP runtime/kernel syntax if supported reliably.
- Implement `BGR2GRAY`.
- Implement `RGB2GRAY`.
- Add CPU reference grayscale conversion for tests.
- Add correctness tests with tiny images first:
  - 1x1
  - 2x2
  - odd width
  - custom `step_bytes`
- Add a larger sanity test such as 640x480.

### Deliverables

- First real GPU image operation.
- CPU reference test coverage.
- Updated example showing upload -> grayscale -> download.

### Exit Criteria

- Grayscale output matches CPU reference within expected integer rounding.
- The operation keeps data on the GPU between input and output.
- The example runs on a Windows machine with AMD HIP SDK.

## Phase 3: Minimal Image Processing MVP

Goal: build enough operations to make `hipcv` useful for real preprocessing
pipelines.

### Function Order

1. `cvtColor`
   - `BGR2GRAY`
   - `RGB2GRAY`
   - `BGR2RGB`
   - `RGB2BGR`
2. `resize`
   - nearest neighbor
   - bilinear
3. `threshold`
   - binary
   - binary inverse
4. `blur`
   - box blur
5. `gaussianBlur`
   - fixed odd kernel sizes first, such as 3x3 and 5x5

### Tasks

- Keep APIs close to OpenCV naming where it improves familiarity.
- Return `Status` instead of throwing exceptions.
- Allocate output matrices internally when needed.
- Validate source format and destination shape consistently.
- Add CPU reference implementations for each operation.
- Add tests for invalid formats and unsupported conversions.
- Add examples for single operations and chained operations.

### Deliverables

- `include/hipcv/imgproc.hpp`
- `src/imgproc.cpp` or HIP-specific source files
- Tests for every MVP operation
- Example pipeline:

```text
upload -> resize -> cvtColor -> threshold -> download
```

### Exit Criteria

- Every MVP operation has at least one correctness test.
- Chained operations do not require intermediate downloads.
- Unsupported operation modes fail clearly with `StatusCode::invalid_argument`
  or a more specific status code if added later.

## Phase 4: Test And CI Foundation

Goal: keep the project trustworthy as GPU kernels are added.

### Tasks

- Add a lightweight C++ test runner or simple test executables.
- Separate tests into:
  - no-HIP tests
  - HIP-required tests
  - CPU reference helpers
- Add CMake options for test categories if needed.
- Add GitHub Actions for no-HIP build checks.
- Keep HIP hardware tests documented as local/manual until GPU CI is available.

### Suggested Test Files

```text
tests/test_status.cpp
tests/test_image_shape.cpp
tests/test_gpu_mat_no_hip.cpp
tests/test_gpu_mat_roundtrip.cpp
tests/test_copy.cpp
tests/test_cvt_color.cpp
tests/test_resize.cpp
tests/test_threshold.cpp
```

### Exit Criteria

- A contributor without AMD hardware can run no-HIP tests.
- A contributor with AMD HIP SDK can run GPU correctness tests.
- CI verifies formatting/build basics without requiring GPU hardware.

## Phase 5: Benchmarking

Goal: measure whether `hipcv` is actually useful, not merely functional.

### Tasks

- Add timing utilities for:
  - upload
  - single operation
  - chained operation
  - download
  - full pipeline
- Test common image sizes:
  - 720p
  - 1080p
  - 4K
- Compare against simple CPU reference implementations first.
- Add optional OpenCV CPU baseline later.
- Measure warm-up separately from steady-state runs.
- Report upload/download overhead separately from GPU compute time.

### Benchmark Pipelines

```text
upload -> download
upload -> copy -> download
upload -> cvtColor -> download
upload -> resize -> cvtColor -> threshold -> download
```

### Deliverables

- `benchmarks/` directory.
- Repeatable benchmark executable.
- Documented sample results for at least one AMD GPU.

### Exit Criteria

- Benchmark output clearly separates transfer cost from compute cost.
- The results show which workflows benefit from staying GPU-resident.

## Phase 6: Developer Experience

Goal: make the project feel like a usable library instead of a kernel demo.

### Tasks

- Improve examples:
  - memory roundtrip
  - grayscale
  - resize
  - chained preprocessing pipeline
- Add clearer error messages.
- Document supported pixel formats and operation modes.
- Document unsupported features explicitly.
- Add a migration guide for OpenCV CUDA users.
- Decide naming conventions before the API grows too much.

### Deliverables

- Better README quickstart.
- `docs/api-design.md`
- `docs/opencv-cuda-migration.md`
- More complete examples.

### Exit Criteria

- A new user can understand the library's scope in under five minutes.
- A C++ user can run one example after following the Windows setup guide.

## Phase 7: Python Binding

Goal: expose the stable C++ core to Python without freezing the wrong API too
early.

### Timing

Start this only after:

- `GpuMat` is stable,
- grayscale works,
- resize works,
- tests exist for the C++ core.

### Tasks

- Add pybind11.
- Expose:
  - `GpuMat`
  - `upload`
  - `download`
  - `cvtColor`
  - `resize`
  - `threshold`
- Support NumPy array input and output.
- Add Python examples.
- Explore Windows wheel packaging.

### Target Python Experience

```python
import hipcv as hcv

gpu = hcv.upload(img)
gray = hcv.cvtColor(gpu, hcv.COLOR_BGR2GRAY)
small = hcv.resize(gray, (640, 360))
out = small.download()
```

### Exit Criteria

- Python can run a full upload -> operation -> download path.
- NumPy array shape and dtype validation are clear.
- Packaging limitations are documented.

## Phase 8: Performance Improvements

Goal: improve real-time pipeline performance after correctness and API shape
are proven.

### Tasks

- Add stream support.
- Add async upload/download.
- Add pinned host memory support.
- Add memory pool or allocator hooks.
- Improve kernel launch configuration.
- Add fused kernels for common pipelines where justified.
- Profile with AMD tools available on Windows.

### Exit Criteria

- Chained operations can run without unnecessary synchronization.
- Benchmarks show measurable gains from stream or memory optimizations.
- Optimization work is guided by profiling, not guesses.

## Phase 9: Compatibility And Expansion

Goal: expand carefully without losing the simple OpenCV-like experience.

### Possible Additions

- More color conversions.
- More resize modes.
- Morphology:
  - `erode`
  - `dilate`
- Edge filters:
  - `sobel`
  - `laplacian`
- Geometric transforms:
  - `warpAffine`
  - `remap`
- Linux ROCm validation.
- Optional integration with AMD libraries if available and useful on Windows.

### Exit Criteria

- New functions follow the same validation, status, and test patterns.
- Windows remains the primary supported path.
- Linux support does not complicate the public API.

## Non-Goals For The MVP

- Reimplementing CUDA.
- Replacing HIP or ROCm.
- DNN inference.
- YOLO runtime integration.
- Full OpenCV API compatibility.
- Supporting every image dtype.
- Building a complex graph execution engine.

## Release Milestones

### v0.1: Technical Scaffold

- Windows-first CMake project.
- Optional HIP detection.
- `GpuMat`.
- `upload`, `download`, `copyTo`.
- Smoke example.
- Basic documentation.

### v0.2: First Kernel

- HIP kernel build path.
- `cvtColor` with `BGR2GRAY` and `RGB2GRAY`.
- CPU reference tests.
- Grayscale example.

### v0.3: Preprocessing MVP

- `resize` nearest and bilinear.
- `threshold`.
- Chained GPU pipeline example.
- Basic benchmarks.

### v0.4: Usability Pass

- Better docs.
- More examples.
- CI no-HIP checks.
- Optional OpenCV CPU comparison.

### v0.5: Python Preview

- pybind11 package experiment.
- NumPy upload/download.
- Python grayscale and resize examples.

## Decision Log To Maintain

Record decisions in docs as they are made:

- Supported Windows and HIP SDK versions.
- HIP kernel compilation strategy.
- Pixel format and dtype support.
- Error handling strategy.
- Testing strategy with and without AMD GPU hardware.
- Whether to use AMD libraries such as RPP for specific operations later.
