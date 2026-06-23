# Validation Log

This file records local build and environment validation results for `hipcv`.

## 2026-06-22: Windows No-HIP Build

### Environment

- OS target selected by CMake: Windows 10.0.26200
- Windows SDK selected by CMake: 10.0.26100.0
- CMake: 4.3.2
- Generator: Visual Studio 18 2026
- Architecture: x64
- C++ compiler: MSVC 19.51.36248.0
- HIP SDK tools: not found on PATH

### Commands

The Codex shell environment had duplicate `PATH` and `Path` variables, which
can make MSBuild fail with:

```text
Key in dictionary: 'Path' Key being added: 'PATH'
```

The validation was run in a temporary child shell with one `Path` variable:

```powershell
cmake --fresh --preset windows-vs2026-no-hip
cmake --build --preset windows-vs2026-no-hip-release
ctest --preset windows-vs2026-no-hip-release --output-on-failure
```

### Result

- Configure: passed
- Build: passed
- Tests: passed

Test result:

```text
1/3 Test #1: hipcv_windows_smoke .............. Passed
2/3 Test #2: hipcv_test_status ................ Passed
3/3 Test #3: hipcv_test_gpu_mat_no_hip ........ Passed
100% tests passed, 0 tests failed out of 3
```

### Notes

- This validates only the no-HIP build path.
- HIP SDK detection and real GPU execution still need to be validated on a
  machine where `hipInfo` and `hipconfig` are available.

## 2026-06-22: Local Development Dependencies

### Installed

- Python: 3.12.13 through `uv python install 3.12`
- Local virtual environment: `.venv`
- Python packages:
  - `numpy==2.5.0`
  - `opencv-python==4.13.0.92`
  - `pybind11==3.0.4`

### Verification

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import numpy, cv2, pybind11; print(numpy.__version__, cv2.__version__, pybind11.__version__)"
```

Result:

```text
Python 3.12.13
numpy 2.5.0
opencv 4.13.0
pybind11 3.0.4
```

### HIP SDK Status

- AMD HIP SDK for Windows is installed at `C:\Program Files\AMD\ROCm\7.1`.
- User environment variables were configured:
  - `HIP_PATH=C:\Program Files\AMD\ROCm\7.1`
  - `ROCM_PATH=C:\Program Files\AMD\ROCm\7.1`
  - `C:\Program Files\AMD\ROCm\7.1\bin` added to the user `Path`

## 2026-06-22: Windows HIP Build

### Environment

- ROCm/HIP SDK path: `C:\Program Files\AMD\ROCm\7.1`
- `hipconfig`: `HIP version: 7.1.51803-d3a86bd04`
- GPU detected by `hipInfo`: AMD Radeon RX 9070 XT
- GPU architecture: `gfx1201`
- Reported global memory: 15.92 GB
- Generator: Visual Studio 18 2026
- Architecture: x64

### Commands

The validation was run in a temporary child shell with one `Path` variable and
explicit ROCm paths:

```powershell
cmake --fresh --preset windows-vs2026
cmake --build --preset windows-vs2026-release
ctest --preset windows-vs2026-release --output-on-failure
```

### Result

- Configure: passed
- Build: passed
- Tests: passed

Test result:

```text
1/3 Test #1: hipcv_windows_smoke .............. Passed
2/3 Test #2: hipcv_test_status ................ Passed
3/3 Test #3: hipcv_test_gpu_mat_no_hip ........ Passed
100% tests passed, 0 tests failed out of 3
```

Smoke executable output:

```text
hipcv 0.1.0
HIP backend: enabled
HIP runtime detected: yes
Device count: 2
HIP runtime version: 70260201
Empty GpuMat: yes
```

### Notes

- `__HIP_PLATFORM_AMD__` must be defined when including HIP headers on this
  Windows HIP SDK setup. The project now defines it for HIP builds.
- Compile-time `.hip` kernel compilation with VS2026/MSVC 14.51 hit a HIP
  Clang/MSVC `<cmath>` overload conflict. The first image kernel therefore uses
  HIPRTC runtime compilation.

## 2026-06-22: First Image Kernel

### Feature

- Added `hipcv::cvtColor`.
- Added `hipcv::ColorConversion::bgr_to_gray`.
- Added `hipcv::ColorConversion::rgb_to_gray`.
- The kernel is compiled with HIPRTC at runtime for the active GPU
  architecture.

### Verification

```powershell
cmake --fresh --preset windows-vs2026
cmake --build --preset windows-vs2026-release
ctest --preset windows-vs2026-release --output-on-failure
build\windows-vs2026\Release\hipcv_cvt_color.exe
```

Result:

```text
1/4 Test #1: hipcv_windows_smoke .............. Passed
2/4 Test #2: hipcv_test_status ................ Passed
3/4 Test #3: hipcv_test_gpu_mat_no_hip ........ Passed
4/4 Test #4: hipcv_test_cvt_color ............. Passed
100% tests passed, 0 tests failed out of 4
BGR2GRAY: 77 149 29
```

### No-HIP Check

The no-HIP preset was also validated after adding `cvtColor`:

```text
100% tests passed, 0 tests failed out of 4
```

## 2026-06-23: Nearest-Neighbor Resize Kernel

### Feature

- Added `hipcv::resize`.
- Added `hipcv::ResizeInterpolation::nearest`.
- Supported initial formats:
  - `gray8`
  - `rgb8`
  - `bgr8`
  - `rgba8`
  - `bgra8`
- The kernel is compiled with HIPRTC at runtime for the active GPU
  architecture.

### Verification

```powershell
cmake --fresh --preset windows-vs2026
cmake --build --preset windows-vs2026-release
ctest --preset windows-vs2026-release --output-on-failure
build\windows-vs2026\Release\hipcv_resize.exe
```

Result:

```text
1/5 Test #1: hipcv_windows_smoke .............. Passed
2/5 Test #2: hipcv_test_status ................ Passed
3/5 Test #3: hipcv_test_gpu_mat_no_hip ........ Passed
4/5 Test #4: hipcv_test_cvt_color ............. Passed
5/5 Test #5: hipcv_test_resize ................ Passed
100% tests passed, 0 tests failed out of 5
resize nearest 3x2 -> 6x4:
1 1 2 2 3 3
1 1 2 2 3 3
4 4 5 5 6 6
4 4 5 5 6 6
```

### No-HIP Check

The no-HIP preset was also validated after adding `resize`:

```text
100% tests passed, 0 tests failed out of 5
```

## 2026-06-23: Binary Threshold Kernel

### Feature

- Added `hipcv::threshold`.
- Added `hipcv::ThresholdType::binary`.
- Added `hipcv::ThresholdType::binary_inv`.
- Initial support is intentionally limited to `gray8`.
- The kernel is compiled with HIPRTC at runtime for the active GPU
  architecture.

### Verification

```powershell
cmake --fresh --preset windows-vs2026
cmake --build --preset windows-vs2026-release
ctest --preset windows-vs2026-release --output-on-failure
build\windows-vs2026\Release\hipcv_threshold.exe
```

Result:

```text
1/6 Test #1: hipcv_windows_smoke .............. Passed
2/6 Test #2: hipcv_test_status ................ Passed
3/6 Test #3: hipcv_test_gpu_mat_no_hip ........ Passed
4/6 Test #4: hipcv_test_cvt_color ............. Passed
5/6 Test #5: hipcv_test_resize ................ Passed
6/6 Test #6: hipcv_test_threshold ............. Passed
100% tests passed, 0 tests failed out of 6
threshold binary:
0 0 0 255
255 255 255 255
```

### No-HIP Check

The no-HIP preset was also validated after adding `threshold`:

```text
100% tests passed, 0 tests failed out of 6
```

## 2026-06-23: Box Blur Kernel

### Feature

- Added `hipcv::blur`.
- Initial support is intentionally limited to `gray8`.
- Kernel sizes must be positive and odd.
- Border handling uses clamp-to-edge.
- The kernel is compiled with HIPRTC at runtime for the active GPU
  architecture.

### Verification

```powershell
cmake --fresh --preset windows-vs2026
cmake --build --preset windows-vs2026-release
ctest --preset windows-vs2026-release --output-on-failure
build\windows-vs2026\Release\hipcv_blur.exe
```

Result:

```text
1/7 Test #1: hipcv_windows_smoke .............. Passed
2/7 Test #2: hipcv_test_status ................ Passed
3/7 Test #3: hipcv_test_gpu_mat_no_hip ........ Passed
4/7 Test #4: hipcv_test_cvt_color ............. Passed
5/7 Test #5: hipcv_test_resize ................ Passed
6/7 Test #6: hipcv_test_threshold ............. Passed
7/7 Test #7: hipcv_test_blur .................. Passed
100% tests passed, 0 tests failed out of 7
blur 3x3:
27 33 43 50
53 60 70 77
80 87 97 103
```

### No-HIP Check

The no-HIP preset was also validated after adding `blur`:

```text
100% tests passed, 0 tests failed out of 7
```

## 2026-06-23: Gaussian Blur Kernel

### Feature

- Added `hipcv::gaussianBlur`.
- Initial support is intentionally limited to `gray8`.
- Supported kernel sizes are 3x3 and 5x5.
- Border handling uses clamp-to-edge.
- The kernel uses fixed binomial Gaussian weights and HIPRTC runtime
  compilation for the active GPU architecture.

### Verification

```powershell
cmake --fresh --preset windows-vs2026
cmake --build --preset windows-vs2026-release
ctest --preset windows-vs2026-release --output-on-failure
build\windows-vs2026\Release\hipcv_gaussian_blur.exe
```

Result:

```text
1/8 Test #1: hipcv_windows_smoke .............. Passed
2/8 Test #2: hipcv_test_status ................ Passed
3/8 Test #3: hipcv_test_gpu_mat_no_hip ........ Passed
4/8 Test #4: hipcv_test_cvt_color ............. Passed
5/8 Test #5: hipcv_test_resize ................ Passed
6/8 Test #6: hipcv_test_threshold ............. Passed
7/8 Test #7: hipcv_test_blur .................. Passed
8/8 Test #8: hipcv_test_gaussian_blur ......... Passed
100% tests passed, 0 tests failed out of 8
gaussianBlur 3x3:
25 33 43 53 60
63 70 80 90 98
113 120 130 140 148
150 158 168 178 185
```

### No-HIP Check

The no-HIP preset was also validated after adding `gaussianBlur`:

```text
100% tests passed, 0 tests failed out of 8
```

## 2026-06-23: Chained Pipeline And Benchmark Harness

### Feature

- Added `hipcv_preprocess_pipeline`.
- Added `hipcv_preprocess_benchmark`.
- Added `HIPCV_BUILD_BENCHMARKS` CMake option.
- The pipeline keeps intermediate data on the GPU:

```text
upload -> resize -> cvtColor -> gaussianBlur -> threshold -> download
```

### Verification

```powershell
cmake --fresh --preset windows-vs2026
cmake --build --preset windows-vs2026-release
ctest --preset windows-vs2026-release --output-on-failure
build\windows-vs2026\Release\hipcv_preprocess_pipeline.exe
build\windows-vs2026\Release\hipcv_preprocess_benchmark.exe
```

Result:

```text
1/8 Test #1: hipcv_windows_smoke .............. Passed
2/8 Test #2: hipcv_test_status ................ Passed
3/8 Test #3: hipcv_test_gpu_mat_no_hip ........ Passed
4/8 Test #4: hipcv_test_cvt_color ............. Passed
5/8 Test #5: hipcv_test_resize ................ Passed
6/8 Test #6: hipcv_test_threshold ............. Passed
7/8 Test #7: hipcv_test_blur .................. Passed
8/8 Test #8: hipcv_test_gaussian_blur ......... Passed
100% tests passed, 0 tests failed out of 8
```

Pipeline output:

```text
preprocess pipeline: upload -> resize -> cvtColor -> gaussianBlur -> threshold -> download
0 0 0 0 0 255 255 255
0 0 0 0 0 255 255 255
0 0 0 0 255 255 255 255
0 0 0 0 255 255 255 255
0 0 0 255 255 255 255 255
0 0 255 255 255 255 255 255
```

Benchmark output on AMD Radeon RX 9070 XT:

```text
hipcv preprocess benchmark
input: 1920x1080 BGR8
pipeline: resize 640x360 -> BGR2GRAY -> gaussianBlur 3x3 -> threshold
iterations: 20

upload H2D                  0.419 ms
GPU pipeline                0.243 ms
download D2H                0.099 ms
```

### No-HIP Check

The no-HIP preset was also validated after adding the pipeline and benchmark
executables:

```text
100% tests passed, 0 tests failed out of 8
```

## 2026-06-23: Supported Operation Matrix And Invalid Argument Tests

### Feature

- Added `docs/supported-operations.md`.
- Added `hipcv_test_imgproc_invalid_args`.
- Documented the MVP-supported image formats, operation modes, and unsupported
  modes.
- Added validation coverage for empty sources, unsupported formats,
  unsupported interpolation/type values, and unsupported blur kernel modes.

### Verification

```powershell
cmake --fresh --preset windows-vs2026
cmake --build --preset windows-vs2026-release
ctest --preset windows-vs2026-release --output-on-failure
```

Result:

```text
1/9 Test #1: hipcv_windows_smoke ............... Passed
2/9 Test #2: hipcv_test_status ................. Passed
3/9 Test #3: hipcv_test_gpu_mat_no_hip ......... Passed
4/9 Test #4: hipcv_test_cvt_color .............. Passed
5/9 Test #5: hipcv_test_resize ................. Passed
6/9 Test #6: hipcv_test_threshold .............. Passed
7/9 Test #7: hipcv_test_blur ................... Passed
8/9 Test #8: hipcv_test_gaussian_blur .......... Passed
9/9 Test #9: hipcv_test_imgproc_invalid_args ... Passed
100% tests passed, 0 tests failed out of 9
```

### No-HIP Check

The no-HIP preset was also validated after adding the invalid argument test:

```text
100% tests passed, 0 tests failed out of 9
```
