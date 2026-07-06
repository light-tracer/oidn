# WebGPU backend — performance notes

Baseline reference: RT filter, 512×512 HDR with albedo+normal, RTX 3060,
steady-state (after warm-up), matching the CPU backend to RMSE ~1e-6.

## Timeline so far

| State | Time | Note |
|-------|------|------|
| Naive scalar conv, one queue submit per op | 1160 ms | |
| OHWI weights + vec4 conv (4 out-channels/thread) | 225 ms | |
| Coalesced (channel-innermost) pool/upsample | 206 ms | |
| Single command buffer + pipeline cache | 21 ms | |
| Fused conv+pool, uniform-buffer arena | ~19 ms | |
| Workgroup-tiled conv (plain convs) | **~20 ms median, ~13 ms best** | current; large run-to-run variance (GPU clock states), medians over 30 runs: 20.5 ms tiled vs 23.4 ms untiled |

The CPU backend (20-thread i7-12700K) does the same denoise in ~63 ms, so the
WebGPU backend is ~3× faster on this machine as of the last measurement.

## What moved the needle (in order of impact)

1. **One command buffer for the whole graph.** Each op used to call
   `wgpuQueueSubmit` individually (~30 submits, ~7 ms each — this dwarfed the
   actual compute). Ops now append compute passes to a shared pending encoder
   (`WebGPUEngine::getEncoder`); `flush()` finishes and submits once. Blocking
   buffer read/write and `sync()` auto-flush to preserve queue ordering.
2. **OHWI weights + vec4 conv kernel.** Input channels contiguous → `vec4`
   loads for both src and weights; each invocation computes 4 output channels,
   amortizing the src loads. Used whenever `ic % 4 == 0` (all interior UNet
   layers); the scalar kernel handles the 9-channel input layers.
3. **Coalesced pool/upsample.** Channel is the innermost HWC dimension, so
   threads iterate channel-first (`gid.x = ox*C + c`).
4. **Pipeline cache** (`WebGPUEngine::getPipeline`). The IO ops were compiling
   their WGSL every submit.

## Tried and rejected (do not retry without new evidence)

- **2×2 pixel register blocking in conv** (each thread computes a 2×2 output
  tile × 4 channels). Breaks x-direction coalescing and raises register
  pressure; measured *slower* than the plain vec4 kernel.
- **16×8 workgroups** for conv. 8×8 is faster on the 3060.

## Implemented since (from the former "ideas" list)

- **Fused conv+pool.** `isConvSupported` now accepts `PostOp::Pool`, so the
  graph builder emits single fused conv+pool ops (the kernel already had the
  `postOp` path): ~4 fewer dispatches and their full-resolution intermediate
  tensors. Fused upsampling stays off — it would recompute each conv result 4×.
  Correctness covered in `test_webgpu_conv.cpp` (fused vs. plain conv + host
  maxpool).
- **Uniform-buffer arena.** Per-dispatch uniform create/release replaced by a
  256 KB bump arena written with `wgpuQueueWriteBuffer`
  (`WebGPUEngine::allocUniform`); slots recycle on wrap-around, which is safe
  because writes are queue-ordered.
- **f16 tensors and weights.** All tensor/weight kernels are WGSL templates
  instantiated for f32 or f16 (`specializeWGSL`, `$T$` placeholder, f32
  accumulation in the conv). When the adapter reports `shader-f16` the device
  requests it and switches `tensorDataType`/`weightDataType` to Float16 —
  halving the bandwidth of the whole network; otherwise everything stays f32
  (same output hash as before, so the refactor is neutral).
  **Caveat:** wgpu-native never exposes SHADER_F16 on NVIDIA Vulkan (wgpu
  gates it on `storageInputOutput16`, which NVIDIA doesn't implement) and its
  DX12 backend would need DXC. So natively this machine runs f32; in Chrome
  (Dawn, DXC) f16 is active — verified LDR + HDR on the demo page, correct
  output. Buffer/binding/copy byte sizes are rounded up to 4 everywhere
  (`roundByteSize4`) because f16 regions can be 2-aligned only (e.g. the
  3-channel bias of the last conv); WebGPU requires multiples of 4.
  The instance now prefers the Vulkan backend natively (with fallback) to
  maximize the chance of f16 on non-NVIDIA GPUs.

- **Workgroup-tiled conv.** Plain (`postOp == None`) convs with `ic % 16 == 0`
  now stage their input tile (8×8 outputs + kernel halo, zero-padded at image
  borders) into workgroup memory in 16-channel chunks
  (`kConv2dTiledWGSL`), so each input pixel is read from global memory once
  per workgroup instead of KH·KW times. Weights stay in global memory (all 64
  threads read the same weight in lockstep — caches/broadcasts fine).
  Measured ~13% faster at the median (20.5 vs 23.4 ms over 30 runs), best
  runs ~13 ms. Not used for fused-pool convs (their tile would be ~4× larger
  than the 16 KB workgroup limit allows per chunk). Note the accumulation
  order changes (channel chunks), so the output hash differs from the vec4
  kernel while staying at RMSE ~1.6e-6 vs the CPU backend.

## Ideas not yet tried (rough expected payoff)

- **Tiled fused-pool conv.** The 4 fused conv+pool layers still use the
  untiled kernel; a tiled variant needs a 17×17 halo tile, so it would have
  to chunk by 8 channels or shrink the output tile.
- **Staging weights in workgroup memory** for the tiled kernel (marginal:
  lockstep reads already broadcast well).
- **1×1 conv fast path.** The 1×1 convs (none in the default RT UNet, but used
  by some models) degenerate to a matmul and could use a wider tiling.

## Web build (Emscripten, browser WebGPU via emdawnwebgpu)

512×512 LDR, RTX 3060, filter cached (see web/oidn_web.cpp), occluded
automated tab: ~400 ms/denoise — upload ~4 ms, execute (record+submit) ~4 ms,
~300 ms waiting on the output readback future. The wait is dominated by the
browser's GPU-process scheduling for hidden tabs; measured pure GPU time was
~70-85 ms, so foreground tabs should be much faster. What mattered:

- **Future-based waits** (`wgpuInstanceWaitAny` + Asyncify) instead of
  `emscripten_sleep` polling. Timer polling was throttled to ~1 s/tick in
  background tabs → ~2 s per denoise (~5× slower).
- **Filter caching in the shim** — recommitting per call re-parses and
  re-reorders the weights in single-threaded wasm (~1 s).
- `executeAsync` + blocking readback (one GPU round trip instead of two).

**Conv pipeline specialization (override constants).** Chrome's Dawn compiles
WGSL through Tint → HLSL → FXC, and FXC generates very slow code for loops
with dynamic (uniform-driven) trip counts — which is what the conv kernel's
kh/kw/ic4 loops were. The vec4 conv pipelines are now specialized per
(IC4, KH, KW) via pipeline-overridable constants (cached per shape, ~10
variants for the UNet), so the loops have compile-time bounds and unroll.
Note the first denoise pays the FXC compilation of all variants (~1.5 s,
one-time); the shim's phase log shows it under "gpu" of the first run.
Native (naga/Vulkan) is unaffected: same output hash, similar timing.

The shim now logs upload / submit / gpu / readback separately (the readback
number is purely the output copy + mapAsync round trip).

**GPU-direct display (no readback).** GPU completion signals awaited by the
wasm (buffer maps, OnSubmittedWorkDone) showed a large systematic latency in
Chrome (~250-900 ms depending on environment) that resisted every workaround
(future waits, JS-side maps, queue kicks, rAF heartbeats) — while the actual
GPU work finished in tens of ms. The demo therefore does not read the result
back at all: the shim publishes the device + output buffer JS objects to the
page (Module.oidnOutput, via the emdawnwebgpu object table), and the page
renders the float3 buffer straight to a WebGPU canvas with a fullscreen pass
that applies the linear→sRGB conversion. Everything is queue-ordered, so no
wait gates the display: a denoise call returns after submission (~6-8 ms) and
the frame presents as soon as the GPU finishes. oidnWebDenoise still supports
the CPU readback path when an output pointer is passed (needed e.g. for
saving the result), which pays the completion-signal latency.

Remaining web levers: the JS float3 input conversion (~40 ms at 512², could
move into wasm or a shader), the completion-signal latency for the readback
path (upstream Chrome/emdawnwebgpu issue worth reporting), and everything in
the native list above (the GPU work itself is the same shaders).

## Profiling aids

- `OIDN_TIME_OPS=1` — prints wall-clock per op in `Graph::submit` (forces a
  device sync per op, so totals are pessimistic on GPU). Native only: on the
  web this path currently aborts; the web shim prints upload/execute/readback
  phase timings to the console instead.
- `OIDN_DUMP_OPS=1` — prints per-op output tensor stats (mean/min/max/nan).
  Invaluable for locating a divergent layer when comparing against the CPU
  backend.
