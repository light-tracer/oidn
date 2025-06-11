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
      -DOIDN_DEVICE_WEBGPU=ON \
      -DOIDN_FILTER_RT=OFF -DOIDN_FILTER_RTLIGHTMAP=OFF .

## Build everything
cmake --build build -j$(nproc)

## Running WebGPU Tests with Software Emulation
WebGPU uses Vulkan by default. In headless environments without a discrete GPU
you can enable Mesa's Lavapipe CPU driver to emulate Vulkan.
Set the `VK_ICD_FILENAMES` environment variable before running tests.
`XDG_RUNTIME_DIR` must also point to a writable directory:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export XDG_RUNTIME_DIR=/tmp/xdg
cd build && ctest --output-on-failure -R WebGPU
```
This forces the Vulkan loader to use Lavapipe so the WebGPU unit tests run
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
Milestone “single-layer bootstrap” is finished:

New device type OIDN_DEVICE_TYPE_WEBGPU should be selectable via public C API.

Source tree:

  devices/webgpu/
  ├─ webgpu_device.h / .cpp      # owns WGPUInstance/Device/Queue
  ├─ webgpu_engine.h / .cpp      # records compute passes
  ├─ webgpu_conv.h / .cpp        # Conv op wrapper
  ├─ webgpu_pool.h / .cpp        # Pool op wrapper
  ├─ webgpu_upsample.h / .cpp    # Upsample op wrapper
  ├─ webgpu_tensor.h             # POD for tensor views
  └─ CMakeLists.txt              # adds option OIDN_DEVICE_WEBGPU

### Implementation details
Raw C API of wgpu-native, no wgpu:: C++ wrapper.
WGSL shaders implement conv2d + bias + ReLU and 2× upsampling.  They live as
inline string literals inside `webgpu_engine.cpp`; no external `.wgsl` files are
shipped.
Fixed stride = 1, no padding, arbitrary N,C,H,W.
Work-group size hard-coded to 8×8×1.
Host ↔ GPU transfers use the public buffer API (`oidnNewBuffer`, `oidnWriteBuffer`,
`oidnReadBuffer`).
Currently kernels for `conv2d_eltwise`, `pool2x2`, `upsample2x`, `add`, `mul`, `softplus`, `input_process`, `output_process`, `image_copy`, and `autoexposure` are implemented in WGSL shaders.
Earlier revisions executed the last four operations on the CPU, but they now run on the GPU as well.
Op classes map these kernels through the standard Engine API and are validated against the CPU backend.

The Metal backend serves as the reference implementation.  It uses NHWC tensor
layout with OIHW weights.  The WebGPU backend now adopts the same layouts and
the tests still compare its results against the CPU backend for convenience.

## Verification Procedure
Build with -DOIDN_DEVICE_WEBGPU=ON.

Ensure environment selects a usable backend
Hardware Vulkan/Metal or Lavapipe SW fallback.

Run: ctest --output-on-failure -R WebGPU
Skipped tests return exit code 77 which CTest recognizes as SKIP.

The tests internally:

* prepare deterministic random tensors,
* execute the kernels using the CPU backend to obtain reference results, and
* run the same kernels on the WebGPU backend.

They pass if
`max(|ref - gpu| / max(|ref|, 1e-6)) < 1e-4`.

## Next Steps / Perspective

Priority	Task	Brief Description
P0      Element-wise kernels    **Done** – WGSL implementations of add, mul, and softplus are available.
P1	Memory allocator	Replace the “one buffer per tensor” strategy with a sub-allocator to reduce memory & improve performance.
P1	Graph execution	Record multiple layers in a single command buffer to amortise overhead.
P2	Full denoiser demo	Make examples/denoise run on a 256×256 tile using the WebGPU backend.
P2	Performance passes	Explore workgroup sizes, shared-memory tiling, fused convolution blocks.
