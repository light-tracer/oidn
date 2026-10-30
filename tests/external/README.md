# Shared-provider tests

This parent CMake project links the minimal static library to the actual Dawn
shipped by Light Tracer. No wgpu-native download, CPU backend, built-in models,
RTLightmap, standalone web wrapper or Asyncify is involved.

Generate CPU fixtures with `tests/webgpu_browser.py` using a separate CPU-enabled
OIDN build. Then configure from a VS developer terminal:

```powershell
cmake -S tests/external -B build-lt -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DLT_ROOT=C:/projects/lighttracer -DOIDN_TEST_WEIGHTS=D:/projects/oidn/weights `
  -DOIDN_TEST_FIXTURES=D:/projects/oidn/build-web-upstream/web/regression-data
cmake --build build-lt
ctest --test-dir build-lt --output-on-failure
```

The native tests cover FP32, strided imported buffers, texture subregions,
alpha/padding, two encoders before submit, destruction before submission, HDR
autoexposure and tiled 1024x1024 execution. The 333x197 LDR buffer test also
records 1100 distinct uniforms into each of two unsubmitted encoders, crossing
the uniform-page boundary. The harness alone reads back results.
Use the executable's final `f16` argument to additionally test a provider with
shader-f16 enabled. The LT native Dawn used here does not advertise that feature.

For the browser use **LT's Emscripten 3.1.74**:

```powershell
C:/projects/emsdk/upstream/emscripten/emcmake.bat cmake -S tests/external -B build-lt-web -G Ninja -DCMAKE_BUILD_TYPE=Release -DLT_ROOT=C:/projects/lighttracer
cmake --build build-lt-web
```

Copy the generated `regression-data/` directory and `weights/rt_ldr.tza` and
`weights/rt_hdr.tza` next to `build-lt-web/index.html`, serve that directory with
`python web/serve.py 8779 --directory build-lt-web`, and click Run. The JavaScript
application creates the only device and submits both encoders. The C++ side
imports that device and uses the same Emscripten resource managers as LT.

The browser runs buffer, texture and in-place variants (24 CPU comparisons),
including distinct C handles referring to the same JavaScript GPU object.
It checks alpha and releases the
application's texture handles immediately after import to exercise manager
reference counts. It requests shader-f16 when available; the FP32 fallback is selected
by the application's device feature list (uncheck Request shader-f16 to test it).
Each transport group uses one application device across all three cases, then
destroys that device to bound peak memory independently of browser GC of released
GPU objects. A small-case-only switch is available for constrained machines.
No hidden blocking read or queue wait
is permitted in library code. Texture conversion buffers are GPU-only and are
reused across frames; direct fusion with network I/O is a future optimization.
The 1024x1024 case changes the memory budget from 512 to 64 MB between encoders
before either is submitted. Optional timestamp queries bracket only OIDN's GPU
commands; warmup and GPU input restoration for in-place cases are excluded.

## Native GPU profiler

Configure a separate build with `-DOIDN_EXTERNAL_PROFILE=ON` in this parent
project, then run:

```powershell
build-lt-profile/oidn_gpu_bench.exe D:/projects/oidn/weights/rt_ldr_calb_cnrm.tza 1024 1024 f32 0 -1 7
```

Arguments: model, width, height, dtype (`f32`/`f16`), kernel policy, memory budget
MB (`-1` is LT's default), warmed repeat count. Policy 0 is optimized; 1 restores
the prior direct/spatial kernels with fused pooling; 2 uses those kernels with
separate pooling. All policies exclude the removed accidental staging copies.
Use `rt_ldr.tza` for color-only or `rt_ldr_calb_cnrm.tza` for two clean guides.
Synthetic inputs are uploaded before measurement. Run 0 is cold and excluded;
RUN lines report GPU graph/convolution time separately from CPU encode, and OP
lines report the last iteration's per-convolution timestamps. This is a speed
harness; CTest and the browser/LT regressions check output correctness.

The test-only instrumentation has no public API and is omitted from normal
library/wasm builds. Timestamp-query support is required. For provider diagnosis,
`OIDN_TEST_VULKAN`, `OIDN_TEST_D3D12`, and `OIDN_TEST_DXC` choose explicit adapter
options; they do not override unsupported features. The pinned LT Dawn has an
explicit NVIDIA Vulkan FP16 blacklist, documented in `devices/webgpu/PERFORMANCE.md`.

## Current upstream Dawn (Windows native)

Set `DAWN_ROOT` to the separate Dawn checkout built in `build-lt` (Release).
This selects modern generated headers and the same DLL used by LT. The legacy
provider and Emscripten paths remain available. Configure `OIDN_TEST_F16=ON` to
register both precisions in CTest (12 cases).

On upstream `08719d35a97`, NVIDIA Vulkan shader-f16 requires driver >=615.71.
The RTX 3060 test host runs 610.88. The optional **test-only** environment variable
`OIDN_TEST_NVIDIA_F16=1` enables upstream's `vulkan_enable_f16_on_nvidia` adapter
toggle for diagnostic parity/profiling. It is not a library option, does not
change Dawn's production guard, and must not be interpreted as supported-driver
validation. With no override this host intentionally uses FP32.
