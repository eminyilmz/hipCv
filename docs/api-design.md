# API Design Notes

`hipcv` is currently a small C++ MVP. The API should stay conservative until
the core GPU memory and image-processing behavior is proven.

## Current Shape

- `GpuMat` owns device memory with move-only RAII semantics.
- Image operations take `const GpuMat& src` and `GpuMat& dst`.
- Operations allocate the destination internally for the MVP.
- Functions return `Status` instead of throwing exceptions.
- HIP-specific details stay behind the public C++ headers.

## Error Handling

The MVP uses explicit `Status` values because it keeps sample code and tests
simple while the public surface is still changing.

| Situation | Expected Status |
| --- | --- |
| Success | `StatusCode::ok` |
| Empty inputs or invalid dimensions | `StatusCode::invalid_argument` |
| Unsupported image format or operation mode | `StatusCode::invalid_argument` |
| Valid GPU operation in a no-HIP build | `StatusCode::hip_not_enabled` |
| GPU allocation failure | `StatusCode::allocation_failed` |
| GPU copy failure | `StatusCode::copy_failed` |
| HIP runtime or HIPRTC failure | `StatusCode::runtime_error` |

## Naming

Public operation names intentionally follow familiar OpenCV-style naming where
that helps users transfer existing knowledge:

- `GpuMat`
- `cvtColor`
- `resize`
- `threshold`
- `blur`
- `gaussianBlur`

The goal is not full OpenCV compatibility. The goal is a practical AMD HIP
image-processing layer with familiar ergonomics.

## Memory Ownership

`GpuMat` owns its device allocation. Copy construction and copy assignment are
deleted to avoid accidental expensive GPU copies. Use `copyTo` for explicit
device-to-device copies.

Future memory features should be added deliberately:

- async copies
- streams
- pinned host memory
- user-owned device pointer views
- submatrix/views

Those features should not leak HIP types into the public API unless there is a
clear interop reason.

## Destination Allocation

MVP operations allocate `dst` internally. This keeps examples concise:

```cpp
hipcv::GpuMat gray;
auto status = hipcv::cvtColor(src, gray, hipcv::ColorConversion::bgr_to_gray);
```

Later versions may add overloads or helpers for preallocated outputs, but the
current behavior should remain predictable:

- validate input first
- allocate destination shape
- run the HIP operation
- release destination on runtime failure

## Python Binding Direction

The future Python layer should feel close to:

```python
gpu = hcv.upload(img)
gray = hcv.cvtColor(gpu, hcv.COLOR_BGR2GRAY)
small = hcv.resize(gray, (640, 360))
out = small.download()
```

The C++ API should avoid choices that make this Python surface awkward.
