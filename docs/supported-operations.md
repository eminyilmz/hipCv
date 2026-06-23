# Supported Operations

This document records the current `hipcv` MVP API surface. It is intentionally
strict: unsupported modes should fail with `StatusCode::invalid_argument`
instead of doing surprising work.

## Image Formats

| Format | Channels | Notes |
| --- | ---: | --- |
| `gray8` | 1 | Main format for threshold and Gaussian blur operations |
| `rgb8` | 3 | Supported by resize, `RGB2GRAY`, and `RGB2BGR` |
| `bgr8` | 3 | Supported by resize, `BGR2GRAY`, and `BGR2RGB` |
| `rgba8` | 4 | Supported by resize |
| `bgra8` | 4 | Supported by resize |

## Operations

| Operation | Supported Input | Supported Modes | Output | Border Handling |
| --- | --- | --- | --- | --- |
| `GpuMat::allocate` | Any positive shape with valid row step | Device allocation | Same shape | N/A |
| `GpuMat::upload` | Host buffer matching shape | Host-to-device copy | Same shape | N/A |
| `GpuMat::download` | Non-empty `GpuMat` | Device-to-host copy | Host buffer | N/A |
| `GpuMat::copyTo` | Non-empty `GpuMat` | Device-to-device copy | Same shape | N/A |
| `cvtColor` | `bgr8`, `rgb8` | `BGR2GRAY`, `RGB2GRAY`, `BGR2RGB`, `RGB2BGR` | `gray8`, `rgb8`, `bgr8` | N/A |
| `resize` | `gray8`, `rgb8`, `bgr8`, `rgba8`, `bgra8` | nearest neighbor, bilinear | Same format, requested size | N/A |
| `threshold` | `gray8` | binary, binary inverse | `gray8` | N/A |
| `blur` | `gray8`, `rgb8`, `bgr8`, `rgba8`, `bgra8` | Positive odd kernel width and height | Same format | Clamp-to-edge |
| `gaussianBlur` | `gray8` | 3x3 and 5x5 square kernels | `gray8` | Clamp-to-edge |

## Current Unsupported Modes

| Area | Not Yet Supported |
| --- | --- |
| `cvtColor` | Alpha formats, grayscale-to-color, HSV/Lab/YUV conversions |
| `resize` | Area, cubic, lanczos |
| `threshold` | Trunc, to-zero, adaptive threshold |
| `blur` | Explicit anchor, border modes |
| `gaussianBlur` | Multi-channel images, arbitrary sigma, broader kernel sizes, explicit border modes |
| Memory | Pinned host memory, async streams, user-owned device pointers |

## Error Behavior

- Empty source matrices return `StatusCode::invalid_argument`.
- Unsupported image formats return `StatusCode::invalid_argument`.
- Unsupported operation modes return `StatusCode::invalid_argument`.
- Valid GPU operations return `StatusCode::hip_not_enabled` when the library is
  built without HIP support.

The MVP uses explicit `Status` values instead of exceptions so callers can keep
error handling predictable while the API is still evolving.
