# hipcv

A HIP-based image processing library that aims to bring an OpenCV CUDA-like
developer experience to AMD GPUs, with Windows as the first target platform.

## Goal

`hipcv` does not try to reimplement CUDA for AMD. Instead, it builds on AMD's
ROCm/HIP ecosystem and provides a higher-level API that should feel familiar to
developers who already use OpenCV.

Target usage:

```python
import hipcv as hcv

gpu = hcv.upload(img)
gray = hcv.cvtColor(gpu, hcv.COLOR_BGR2GRAY)
blur = hcv.gaussianBlur(gray, (5, 5), 1.2)
out = blur.download()
```

## Initial MVP Scope

- `GpuMat`
- `upload` / `download`
- `Stream`
- `cvtColor`
- `resize`
- `threshold`
- `blur`
- `gaussianBlur`
- Python binding
- Accuracy tests against OpenCV CPU output
- Basic benchmark tooling

## Market Gap

AMD already provides strong building blocks such as ROCm, HIP SDK for Windows,
RPP, rocAL, and MIVisionX. However, there is still no obvious,
general-purpose, Python-friendly AMD GPU image processing layer that feels as
direct as OpenCV CUDA's `GpuMat` and `cv::cuda` workflow, especially for
Windows application developers.

`hipcv` focuses on that gap:

- Hide the low-level complexity of HIP behind a practical API.
- Provide OpenCV-like function names and data flow.
- Reuse existing AMD components such as ROCm/RPP where it makes sense.
- Optimize for real-time pipelines, not just isolated function calls.

## Initial Target Platform

- Windows 11 x86-64
- AMD HIP SDK for Windows
- ROCm-supported AMD Radeon or Radeon PRO GPU
- Visual Studio 2022 or compatible C++ toolchain
- CMake
- C++17
- Python 3.10+
- OpenCV CPU baseline

Linux support is still relevant, but it is no longer the first milestone.

## Windows Notes

The Windows build will target AMD HIP SDK rather than the full Linux ROCm
stack. This matters because the Windows SDK exposes a subset of ROCm, and some
ROCm components available on Linux are not available on Windows.

Before running `hipcv`, users should be able to run:

```powershell
hipInfo
hipconfig
```

See [Windows development notes](docs/windows-development.md) for the planned
setup and constraints.

## Secondary Platform

- Linux
- ROCm-supported AMD GPU
- C++17
- HIP
- Python 3.10+
- OpenCV CPU baseline

## Project Status

This repository is in the early design and MVP planning stage.
