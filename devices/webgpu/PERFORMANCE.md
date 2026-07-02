# WebGPU backend — performance notes

Baseline reference: RT filter, 512×512 HDR with albedo+normal, RTX 3060,
steady-state (after warm-up), matching the CPU backend to RMSE ~1e-6.

## Timeline so far

| State | Time | Note |
|-------|------|------|
| Naive scalar conv, one queue submit per op | 1160 ms | |
| OHWI weights + vec4 conv (4 out-channels/thread) | 225 ms | |
| Coalesced (channel-innermost) pool/upsample | 206 ms | |
| Single command buffer + pipeline cache | **21 ms** | current |

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

## Ideas not yet tried (rough expected payoff)

- **Shared-memory / workgroup tiled conv.** The conv kernel currently reads src
  and weights straight from global memory every MAC. It runs at ~250 GFLOP/s,
  which is ~2% of the 3060's FP32 peak — memory-bound, so a classic tiled conv
  (stage an input tile + the 3×3 weights into `var<workgroup>` and reuse across
  the workgroup) is the highest-upside remaining lever. Could plausibly get the
  denoise into the sub-10 ms range.
- **f16 weights and tensors.** OIDN's other GPU backends denoise in half
  precision. Halves bandwidth (the bottleneck) and enables `shader-f16`.
  Requires the `shader-f16` feature (widely available) and packing weights as
  f16; the network tolerates it (that is what the .tza large weights are).
- **Fused conv post-ops.** The conv kernel already implements fused
  Pool/Upsample (`postOp` field), but `WebGPUEngine::isConvSupported` still
  reports `None` only, so the graph builder splits every pooled/upsampled conv
  into two ops (two dispatches, an extra intermediate tensor). Overriding
  `isConvSupported` to accept `Pool`/`Upsample` would remove ~8 dispatches and
  their intermediates. Needs a fused-conv correctness test first.
- **Persistent per-op uniform buffers.** Every dispatch allocates a small
  uniform buffer for its params and releases it. Pooling/reusing these would
  cut allocator churn (minor, but free).
- **1×1 conv fast path.** The 1×1 convs (none in the default RT UNet, but used
  by some models) degenerate to a matmul and could use a wider tiling.

## Profiling aids (both work on any backend)

- `OIDN_TIME_OPS=1` — prints wall-clock per op in `Graph::submit` (forces a
  device sync per op, so totals are pessimistic on GPU).
- `OIDN_DUMP_OPS=1` — prints per-op output tensor stats (mean/min/max/nan).
  Invaluable for locating a divergent layer when comparing against the CPU
  backend.
