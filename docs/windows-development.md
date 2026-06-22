# Windows Development Notes

`hipcv` is now planned as a Windows-first project.

## Target Environment

- Windows 11 x86-64
- AMD HIP SDK for Windows
- Supported AMD Radeon or Radeon PRO GPU
- Visual Studio 2022 or compatible C++ build tools
- CMake
- Python 3.10+
- OpenCV CPU baseline for correctness tests

## Why Windows First?

Many OpenCV users build camera, video, automation, and desktop vision tools on
Windows. The goal is to make AMD GPU acceleration feel approachable in that
environment without asking users to move their entire workflow to Linux.

## Important Constraint

HIP SDK for Windows is a subset of the full ROCm platform. The initial design
should therefore depend only on components that are available on Windows:

- HIP runtime
- HIP compiler/tooling
- HIP math and primitive libraries when available
- Windows-compatible profiling and debugging tools

AI libraries and some Linux ROCm components should be treated as out of scope
for the first MVP.

## First Technical Milestone

The first milestone is not a full image processing library. It is a minimal
Windows proof of life:

1. Detect HIP SDK from CMake.
2. Build a tiny HIP kernel on Windows.
3. Allocate device memory.
4. Upload a small image buffer.
5. Run a simple grayscale or copy kernel.
6. Download the result.
7. Compare the output with OpenCV CPU code.

## Expected Validation Commands

Users should first verify the HIP SDK installation:

```powershell
hipInfo
hipconfig
```

Then the project provides a Windows smoke test:

```powershell
cmake --preset windows-vs2022
cmake --build --preset windows-vs2022-release
ctest --preset windows-vs2022-release
```

If HIP SDK is not installed yet, the project should still configure and build
without the GPU backend. In that mode, the smoke test reports that HIP is
disabled. To force a documentation or CPU-only build:

```powershell
cmake --preset windows-vs2022-no-hip
cmake --build --preset windows-vs2022-no-hip-release
ctest --preset windows-vs2022-no-hip-release
```

The HIP SDK finder checks common Windows locations such as:

```text
C:\Program Files\AMD\ROCm\*
C:\Program Files\AMD\HIP SDK\*
```

It also honors these environment variables when present:

```text
HIP_PATH
ROCM_PATH
HIPSDK_ROOT
ROCM_ROOT
```

## Design Rule

Platform-specific code should stay isolated. The public API should not expose
Windows-specific details unless there is no reasonable alternative.
