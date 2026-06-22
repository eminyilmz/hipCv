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
