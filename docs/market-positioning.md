# Market Positioning

## Short Positioning

An easy-to-use, Python-friendly image processing layer that brings an OpenCV
CUDA-like workflow to AMD GPUs.

## What It Is Not

- Not a CUDA clone.
- Not a replacement for ROCm/HIP.
- Not a DNN inference framework in the first stage.
- Not a full OpenCV replacement.

## What It Is

- A higher-level image processing API built on HIP.
- Familiar functions for developers who already use OpenCV.
- A way to chain operations while data stays on the AMD GPU.
- A small, reliable core backed by benchmarks and accuracy tests.

## Difference from Existing Solutions

| Solution | Strength | hipcv Difference |
| --- | --- | --- |
| ROCm/HIP | Low-level GPU programming | Higher-level OpenCV-like API |
| OpenCV CUDA | Mature and convenient CUDA workflow | AMD GPU target |
| OpenCV UMat/OpenCL | Portability | More explicit HIP backend and performance control |
| RPP | Performance primitives | Simpler application developer API |
| rocAL | ML data pipelines | General-purpose image processing |
| MIVisionX | Comprehensive OpenVX toolkit | Lighter OpenCV/Python-focused experience |
