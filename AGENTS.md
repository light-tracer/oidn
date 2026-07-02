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

### Windows build (added 2026-07)
Works with VS2022 + Ninja. Prerequisites: ISPC binary + oneTBB binary release,
weights + mkl-dnn submodules (NOTE: .gitmodules relative URLs resolve to the
light-tracer org which lacks the submodule repos; override with
`git config submodule.mkl-dnn.url https://github.com/OpenImageDenoise/mkl-dnn.git`
and `git config submodule.weights.url https://github.com/OpenImageDenoise/oidn-weights.git`).

```
git submodule update --init weights external/mkl-dnn
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_WEBGPU=ON ^
      -DISPC_EXECUTABLE=<deps>/ispc-v1.26.0-windows/bin/ispc.exe ^
      -DTBB_ROOT=<deps>/oneapi-tbb-2021.12.0
cmake --build build -j
```
Run from a VS x64 developer environment (vcvars64.bat). At runtime, the TBB
redist dir and the build dir (wgpu_native.dll) must be on PATH.
wgpu-native Windows asset (x86_64-msvc) is wired into cmake/oidn_wgpu.cmake.

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
The only presistent thing is commited code. While iterating on code feel free to add special comments like:

// AICODE-NOTE: info useful to keep between iterations or sessions
// AICODE-TODO: mini-tasks for agent(s) to keep track of between iterations or sessions
// AICODE-ASK: open questions stored for later or for human to answer

## WebGPU Backend – Current State ✅
Milestone **“end-to-end RT denoiser”** is finished (2026-07):

* `oidnDenoise -d wgpu -f RT --hdr color.pfm --alb albedo.pfm --nrm normal.pfm`
  produces output matching the CPU backend to RMSE ~1e-6 (512×512, 1 tile,
  HDR with autoexposure, and LDR; with and without aux images).
* The conv kernel implements the core Conv contract: same-padding convolution
  (dst H/W == src H/W), configurable activation (none/ReLU), per-channel or
  full-tensor bias, and (dormant) fused Pool/Upsample post-ops. `ConcatConvHWC`
  (UNet skip connections) works on top of it — the second conv's dst-as-bias
  aliasing is handled with a special bias mode that reads the bias through the
  dst binding (WebGPU forbids binding one buffer as both read-only and
  read-write in a dispatch).
* Graph execution notes (hard-won, do not regress):
  - wgpu tracks buffer usage per-buffer, not per-range. All tensors of a graph
    are suballocated from one heap `WGPUBuffer`, so **all** tensor bindings in
    the conv/pool/upsample/eltwise kernels are `read_write` storage.
  - Binding offsets must include BOTH the `WebGPUBuffer`'s own offset within
    the heap (`getWGPUBufferOffset()`) and the tensor's offset within the
    buffer (`Memory::getByteOffset()`). A missing `.offset` in the pool bind
    group entries silently corrupted the whole UNet at one point.
  - `WebGPUEngine::getBufferByteSizeAndAlignment` returns 256-byte alignment
    (WebGPU minStorageBufferOffsetAlignment).
  - Images (filter inputs/outputs) are accessed via `WebGPUImageAccess`
    (webgpu_image_util.h): device-buffer-backed images are bound directly,
    host-memory images are staged. Never call `Image::getPtr()` for device
    images — it is null (no unified memory in WebGPU!).
  - `Autoexposure::getDstPtr()` is virtual now; the WebGPU op host-mediates the
    result (the core hands that pointer to TransferFunction, which dereferences
    it on the host — a UMA assumption that WebGPU cannot satisfy).
  - The autoexposure kernel replicates the CPU binned log-average algorithm.
  - Debug aid: set `OIDN_DUMP_OPS=1` to print per-op output stats (mean/min/
    max/nan) during Graph::submit on any backend — invaluable for comparing
    backends layer by layer.

Earlier milestone “single-layer bootstrap”:

New device type OIDN_DEVICE_TYPE_WEBGPU should be selectable via public C API.

Source tree:

  devices/webgpu/
  ├─ webgpu_device.h / .cpp      # owns WGPUInstance/Device/Queue
  ├─ webgpu_engine.h / .cpp      # records compute passes
  ├─ webgpu_conv.h / .cpp        # Conv op wrapper
  ├─ webgpu_pool.h / .cpp        # Pool op wrapper
  ├─ webgpu_upsample.h / .cpp    # Upsample op wrapper
  ├─ webgpu_tensor.h             # POD for tensor views
  ├─ webgpu_buffer.h / .cpp      # device buffer implementation
  ├─ webgpu_heap.h / .cpp        # heap backing arenas
  ├─ webgpu_arena.h / .cpp       # simple allocator built on WebGPUHeap
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
Memory is managed through `WebGPUArena` / `WebGPUHeap`, which allow sub-allocating
multiple buffers from a single WebGPU buffer.
Currently kernels for `conv2d_eltwise`, `pool2x2`, `upsample2x`, `add`, `mul`, `softplus`, `input_process`, `output_process`, `image_copy`, and `autoexposure` are implemented in WGSL shaders.
Earlier revisions executed the last four operations on the CPU, but they now run on the GPU as well.
Op classes map these kernels through the standard Engine API and are validated against the CPU backend where a matching CPU implementation exists.

The Metal backend serves as the reference implementation.  It uses NHWC tensor
layout with OIHW weights.  The WebGPU backend now adopts the same layouts and
the tests still compare its results against the CPU backend for convenience.
The `test_webgpu_conv` unit test has been updated to compute its reference
output using the CPU backend instead of a hand-written loop.
CPU reference functions require calling `setTile()` to cover the full
image, otherwise the default tile size of zero results in an all-zero
output.  The WebGPU tests set the tile accordingly before launching both
the CPU and GPU kernels.

## Verification Procedure
Build with -DOIDN_DEVICE_WEBGPU=ON.

Ensure environment selects a usable backend
Hardware Vulkan/Metal or Lavapipe SW fallback.

Run: ctest --output-on-failure -R WebGPU
Skipped tests return exit code 77 which CTest recognizes as SKIP.

`WebGPU.Autoexposure` runs successfully after fixing a lifetime issue
with the CPU reference buffer. `WebGPU.Eltwise` is still skipped because
the CPU backend lacks dedicated element-wise kernels and the WebGPU
implementation sometimes produces non-deterministic results on Lavapipe.
`WebGPU.EltwiseMetal` attempts to compare the kernels against the Metal
backend but is skipped unless that backend is available.

`WebGPU.Arena` exercises the buffer heap allocator.

The tests internally:

* prepare deterministic random tensors,
* execute the kernels using the CPU backend to obtain reference results, and
* run the same kernels on the WebGPU backend.

They pass if
`max(|ref - gpu| / max(|ref|, 1e-6)) < 1e-4`.

## Next Steps / Perspective

Priority	Task	Brief Description
P0      Element-wise kernels    **Done** – WGSL implementations of add, mul, and softplus are available.
P1      Memory allocator        **Done** – buffers share memory through WebGPUArena/WebGPUHeap.
P1	Full denoiser	**Done** – oidnDenoise runs the RT filter on WebGPU, matching CPU output (RMSE ~1e-6).
P1	Performance	**Done (first pass)** – 512×512 HDR denoise: 1.16 s → 21 ms steady-state on an RTX 3060 (vs 63 ms on the 20-thread CPU backend). What worked, in order of impact: (1) whole graph recorded into ONE command buffer via WebGPUEngine::getEncoder()/flush() — per-op wgpuQueueSubmit was ~7 ms each, by far the dominant cost; buffer read/write and sync() flush automatically to keep queue ordering; (2) OHWI weight layout + vec4 conv kernel computing 4 output channels per thread (5x on conv time); (3) channel-innermost (coalesced) pool/upsample kernels; (4) pipeline cache (WebGPUEngine::getPipeline) so IO ops stop recompiling WGSL every submit. What did NOT work (tried and reverted): 2x2 pixel register blocking in conv (breaks x-coalescing, register pressure), 16x8 workgroups (8x8 is better). Next levers if needed: shared-memory tiled conv (conv is at ~250 GFLOP/s, ~2% of peak), fused conv post-ops (kernel implements them; isConvSupported still reports None), f16 weights/tensors.
P2	Web/Emscripten build	Nothing exists yet: cmake fetches native wgpu-native binaries (linux/macos/windows). A web build needs emdawnwebgpu/Emscripten plumbing and an async entry point (no blocking waits on the web).
P2	Robustness	Image row strides are assumed packed float3; tiling >1 tile untested on WebGPU; WebGPU.Eltwise still skipped (non-determinism seen on Lavapipe only).
