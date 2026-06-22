# hipcv

A HIP-based image processing library that aims to bring an OpenCV CUDA-like
developer experience to AMD GPUs.

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

AMD already provides strong building blocks such as ROCm, HIP, RPP, rocAL, and
MIVisionX. However, there is still no obvious, general-purpose, Python-friendly
AMD GPU image processing layer that feels as direct as OpenCV CUDA's `GpuMat`
and `cv::cuda` workflow.

`hipcv` focuses on that gap:

- Hide the low-level complexity of HIP behind a practical API.
- Provide OpenCV-like function names and data flow.
- Reuse existing AMD components such as ROCm/RPP where it makes sense.
- Optimize for real-time pipelines, not just isolated function calls.

## Initial Target Platform

- Linux
- ROCm-supported AMD GPU
- C++17
- HIP
- Python 3.10+
- OpenCV CPU baseline

Windows support will be evaluated later.

## Project Status

This repository is in the early design and MVP planning stage.
