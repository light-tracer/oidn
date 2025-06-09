# OIDN Development Notes – Agent Edition

The document is meant to be read by autonomous coding agents.  
It tells you **what already works**, **how to reproduce it** and **next steps** to give you context and general direction.
Follow the user prompt to determine what to work on.

## Project Purpose
Intel Open Image Denoise (OIDN) is a library for removing noise from ray traced images. It contains multiple device backends (CPU, SYCL, CUDA, HIP, Metal) which implement the same denoising API.

End Goal: add the new backend based on the WebGPU API.

---

## 0. Fresh Session

## Configure: enable WebGPU backend (OFF by default)
cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOIDN_DEVICE_WEBGPU=ON .

## Build everything
cmake --build build -j$(nproc)

## Running WebGPU Tests with Software Emulation
WebGPU uses Vulkan by default. In headless environments without a discrete GPU
you can enable Mesa's Lavapipe CPU driver to emulate Vulkan.
Set the `VK_ICD_FILENAMES` environment variable before running tests:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
cd build && ctest --output-on-failure -R WebGPU.Conv2d
```
This forces the Vulkan loader to use Lavapipe so the WebGPU unit test runs
entirely in software.

### WebGPU
- Prebuilt `wgpu-native` binaries are downloaded during configure from
  `https://github.com/gfx-rs/wgpu-native` (Linux x86_64 release). The CMake macro
  `oidn_download_wgpu()` in `cmake/oidn_wgpu.cmake` fetches and unpacks the
  archive to the build tree.
- Currently version `v25.0.2.1` is used.
- CMake automatically downloads a prebuilt `wgpu-native` package when
  `OIDN_DEVICE_WEBGPU` is enabled. No manual installation of the library is
  required.
- The legacy sample programs under `devices/wgpu` are kept for reference but
  are not used by the unit tests.

## Environment Persistence
This workspace is ephemeral. Packages installed with `apt-get`, downloaded weights, and built artifacts vanish after the session ends.

## WebGPU Backend – Current State ✅
Milestone “single-layer bootstrap” is almost finished:

New device type OIDN_DEVICE_TYPE_WEBGPU should be selectable via public C API.

Source tree:

devices/webgpu/
  ├─ webgpu_device.h / .cpp      # owns WGPUInstance/Device/Queue
  ├─ webgpu_engine.h / .cpp      # records 1 compute pass
  ├─ webgpu_tensor.h             # POD for tensor views
  └─ CMakeLists.txt              # adds option OIDN_DEVICE_WEBGPU

### Implementation details
Raw C API of wgpu-native, no wgpu:: C++ wrapper.
WGSL for conv2d + Bias + ReLU lives as an inline string literal
inside webgpu_engine.cpp; no external .wgsl file is shipped.
Fixed stride = 1, no padding, arbitrary N,C,H,W.
Work-group size hard-coded to 8×8×1.
Host ↔ GPU transfers: naïve per-tensor buffers (CreateBuffer + Map).

## Verification Procedure
Build with -DOIDN_DEVICE_WEBGPU=ON.

Ensure environment selects a usable backend
Hardware Vulkan/Metal or Lavapipe SW fallback.

Run: ctest --output-on-failure -R WebGPU.Conv2d

The test internally:

prepares deterministic random tensors,
runs the exact same layer on CPU & WebGPU,
declares success if
max( |ref - gpu| / max(|ref|, 1e-6) ) < 1e-4.

## Next Steps / Perspective

Priority	Task	Brief Description
P0	Public buffer API	Support oidnNewBuffer / oidnReadBuffer so users can upload & download tensors without the private engine helpers.
P0	More primitive kernels	Implement pooled, upsample, element-wise, softplus, etc. Follow the same inline-WGSL approach.
P1	Memory allocator	Replace the “one buffer per tensor” strategy with a sub-allocator to reduce memory & improve performance.
P1	Graph execution	Record multiple layers in a single command buffer to amortise overhead.
P2	Full denoiser demo	Make examples/denoise run on a 256×256 tile using the WebGPU backend.
P2	Performance passes	Explore workgroup sizes, shared-memory tiling, fused convolution blocks.