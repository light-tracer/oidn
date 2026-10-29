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
autoexposure and tiled 1024x1024 execution. The harness alone reads back results.
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

The browser runs both buffer and texture cases, checks alpha, and releases the
application's texture handles immediately after import to exercise manager
reference counts. It requests shader-f16 when available; the FP32 fallback is selected
by the application's device feature list. No hidden blocking read or queue wait
is permitted in library code. Texture conversion buffers are GPU-only and are
reused across frames; direct fusion with network I/O is a future optimization.
