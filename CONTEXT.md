# OIDN Development Notes

## Project Purpose
Intel Open Image Denoise (OIDN) is a library for removing noise from ray traced images. It contains multiple device backends (CPU, SYCL, CUDA, HIP, Metal) which implement the same denoising API.

## Backend Selection Overview
- Device modules register themselves via `Context::registerDeviceType`. Each module detects available physical devices and provides a `DeviceFactory` implementation.
- `Context::init()` loads device modules based on `OIDN_DEVICE_*` environment variables and builds a list of physical devices.
- `oidnNewDevice` or `oidnNewDeviceByID` constructs a `Device` using this registry. The `OIDN_DEFAULT_DEVICE` environment variable can override the default device type or physical device ID.

## CPU Backend Dependencies
To build the CPU backend on Ubuntu, install ISPC and oneTBB:
```bash
apt-get update
apt-get install -y libtbb-dev libtbb12 ispc
```
Configure and build with CMake/Ninja (weights optional if not fetched):
```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DOIDN_FILTER_RT=OFF \
      -DOIDN_FILTER_RTLIGHTMAP=OFF ..
ninja -j4
```

## Weights Submodule
The pretrained filter weights live in the `weights` submodule and use Git LFS. To fetch them:
```bash
git submodule update --init --recursive weights
cd weights && git lfs pull
```
Without Git LFS the blobs will be invalid and the build fails.

## Metal Backend Highlights
- Implemented under `devices/metal/` using Objective‑C++ and MPSGraph.
- `MetalDevice` enumerates available `MTLDevice` objects and manages a command queue.
- `MetalEngine` submits compute kernels via Metal compute pipelines and MPSGraph ops for convolutions.
- Module registration occurs in `metal_module.mm` which calls `Context::registerDeviceType`.

## WebGPU (Dawn) Backend – Implementation Plan
Goal: add a backend based on the Dawn WebGPU library. First milestone is a minimal `wgpu` backend executing a simple identity kernel.

1. **New Device Type**
   - Extend `DeviceType` enum and API enums to include `OIDN_DEVICE_TYPE_WGPU`.
   - Add build option `OIDN_DEVICE_WGPU` in CMake similar to other device toggles.
2. **Backend Skeleton**
   - Create `devices/wgpu` directory with `CMakeLists.txt` and source files implementing `WGPUDevice`, `WGPUEngine`, and a registration module.
   - `WGPUDevice` detects adapters via Dawn, selects one, and owns a command queue.
   - `WGPUEngine` wraps WebGPU command buffers and pipelines; provide minimal buffer and tensor classes.
3. **Identity Kernel Prototype**
   - Add a `.wgsl` shader implementing an identity copy of an image buffer.
   - Compile the WGSL during build or at runtime using Dawn utilities.
   - Implement a simple op in `WGPUEngine::submitKernel` dispatching this shader to validate the pipeline.
4. **Build Integration**
   - Link against Dawn libraries (dawn_wire, dawn_native). Provide notes to install Dawn or fetch a prebuilt binary.
   - Ensure CPU and other device builds remain unaffected when `OIDN_DEVICE_WGPU=OFF` (default).
5. **Testing**
   - Add a minimal unit test or example to run the identity kernel and verify output.

### Implementation Notes
- Prebuilt `wgpu-native` binaries are downloaded during configure from
  `https://github.com/gfx-rs/wgpu-native` (Linux x86_64 release). The CMake macro
  `oidn_download_wgpu()` in `cmake/oidn_wgpu.cmake` fetches and unpacks the
  archive to the build tree.
- The `devices/wgpu` directory contains a small `wgpuIdentity` program using the
  pure C WebGPU API. It dispatches a WGSL shader copying a buffer to another and
  prints `PASSED` if the copied data matches.
- `scripts/test.py` runs this program when invoked with `--device wgpu`.


## Environment Persistence
This workspace is ephemeral. Packages installed with `apt-get`, downloaded weights, and built artifacts vanish after the session ends. Reinstall dependencies and run `git lfs pull` each time a new session starts.
