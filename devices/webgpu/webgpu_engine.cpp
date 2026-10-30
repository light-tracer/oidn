#include "webgpu_engine.h"
#include "webgpu_compat.h"
#include "webgpu_device.h"
#include "webgpu_buffer.h"
#include "webgpu_heap.h"
#include "webgpu_conv.h"
#include "webgpu_pool.h"
#include "webgpu_upsample.h"
#include "webgpu_input_process.h"
#include "webgpu_output_process.h"
#include "webgpu_image_copy.h"
#include "webgpu_autoexposure.h"
#include "webgpu_texture.h"
#include "core/concat_conv.h"
#include <cmath>
#include <cstring>

OIDN_NAMESPACE_BEGIN

  // Same-padding convolution (matching the core Conv contract: dst H/W = src H/W
  // for PostOp::None), with optional fused ReLU, 2x2 max pooling or 2x nearest
  // upsampling, and three bias modes:
  //   0 = per-channel bias vector
  //   1 = full-tensor bias (same shape as the conv output)
  //   2 = full-tensor bias read through the dst binding (used by ConcatConvHWC,
  //       where dst doubles as bias; binding the same buffer as both read-only
  //       and read-write would violate WebGPU aliasing rules)
  // $T$ is the tensor element type (f32/f16); accumulation is always f32.
  static const char* kConv2dWGSL = R"wgsl(
  struct Tensor { data: array<$T$>, };

  @group(0) @binding(0) var<storage, read_write> src   : Tensor;
  @group(0) @binding(1) var<storage, read_write> weight: Tensor;
  @group(0) @binding(2) var<storage, read_write> bias  : Tensor;
  @group(0) @binding(3) var<storage, read_write> dst: Tensor;

  struct Params { ic: u32, ih: u32, iw: u32,
                  oc: u32, oh: u32, ow: u32,
                  kh: u32, kw: u32,
                  padH: u32, padW: u32,
                  activation: u32, // 0=none, 1=relu
                  biasMode: u32,   // 0=per-channel, 1=tensor, 2=tensor aliased with dst
                  postOp: u32,     // 0=none, 1=pool2x2, 2=upsample2x
                  pad0: u32, pad1: u32, pad2: u32, };
  @group(0) @binding(4) var<uniform> p: Params;

  // 4 consecutive output channels at position (cy, cx) of the same-padded
  // conv output grid (ih x iw). Weights are in OHWI layout. Out-of-range
  // output channels are computed with clamped weight reads and discarded by
  // the caller.
  fn conv4_at(cy: i32, cx: i32, oc0: u32) -> vec4<f32> {
    let wStride = p.kh * p.kw * p.ic; // between consecutive output channels
    var acc = vec4<f32>(0.0);
    for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
      let iy = cy + i32(ky) - i32(p.padH);
      if (iy < 0 || iy >= i32(p.ih)) { continue; }
      for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
        let ix = cx + i32(kx) - i32(p.padW);
        if (ix < 0 || ix >= i32(p.iw)) { continue; }
        let sBase = (u32(iy) * p.iw + u32(ix)) * p.ic;
        let wBase = ((oc0 * p.kh + ky) * p.kw + kx) * p.ic;
        for (var ic: u32 = 0u; ic < p.ic; ic = ic + 1u) {
          let s = f32(src.data[sBase + ic]);
          acc.x = acc.x + s * f32(weight.data[wBase + ic]);
          acc.y = acc.y + s * f32(weight.data[wBase + wStride + ic]);
          acc.z = acc.z + s * f32(weight.data[wBase + 2u * wStride + ic]);
          acc.w = acc.w + s * f32(weight.data[wBase + 3u * wStride + ic]);
        }
      }
    }
    return acc;
  }

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc0 = gid.z * 4u;
    if (ox >= p.ow || oy >= p.oh || oc0 >= p.oc) { return; }

    var acc: vec4<f32>;
    if (p.postOp == 1u) {
      // max pooling commutes with the bias add and (monotonic) ReLU
      let y0 = i32(oy * 2u);
      let x0 = i32(ox * 2u);
      var m = conv4_at(y0, x0, oc0);
      m = max(m, conv4_at(y0, x0 + 1, oc0));
      m = max(m, conv4_at(y0 + 1, x0, oc0));
      m = max(m, conv4_at(y0 + 1, x0 + 1, oc0));
      acc = m;
    } else if (p.postOp == 2u) {
      acc = conv4_at(i32(oy / 2u), i32(ox / 2u), oc0);
    } else {
      acc = conv4_at(i32(oy), i32(ox), oc0);
    }

    let dstBase = (oy * p.ow + ox) * p.oc + oc0;
    let n = min(4u, p.oc - oc0);
    for (var j: u32 = 0u; j < n; j = j + 1u) {
      var v = acc[j];
      if (p.biasMode == 0u) {
        v = v + f32(bias.data[oc0 + j]);
      } else if (p.biasMode == 1u) {
        v = v + f32(bias.data[dstBase + j]);
      } else {
        v = v + f32(dst.data[dstBase + j]);
      }
      if (p.activation == 1u && v < 0.0) { v = 0.0; }
      dst.data[dstBase + j] = $T$(v);
    }
  }
  )wgsl";

  // Vectorized variant used when the input channel count is a multiple of 4:
  // src and weight (OHWI) are read as vec4s, and each invocation computes 4
  // consecutive output channels, amortizing the src loads.
  //
  // The loop bounds are pipeline-overridable constants (one pipeline per
  // (IC4, KH, KW) combination, cached): with compile-time trip counts the
  // loops fully unroll, which matters enormously for browsers where Tint
  // compiles WGSL through HLSL/FXC (dynamic loops there are very slow).
  static const char* kConv2dVec4WGSL = R"wgsl(
  struct TensorF { data: array<$T$>, };
  struct TensorV { data: array<vec4<$T$>>, };

  @group(0) @binding(0) var<storage, read_write> src   : TensorV;
  @group(0) @binding(1) var<storage, read_write> weight: TensorV;
  @group(0) @binding(2) var<storage, read_write> bias  : TensorF;
  @group(0) @binding(3) var<storage, read_write> dst   : TensorF;

  struct Params { ic: u32, ih: u32, iw: u32,
                  oc: u32, oh: u32, ow: u32,
                  kh: u32, kw: u32,
                  padH: u32, padW: u32,
                  activation: u32,
                  biasMode: u32,
                  postOp: u32,
                  pad0: u32, pad1: u32, pad2: u32, };
  @group(0) @binding(4) var<uniform> p: Params;

  override IC4: u32 = 8u; // input channels / 4
  override KH: u32 = 3u;
  override KW: u32 = 3u;

  // 4 consecutive output channels (oc0..oc0+3) at conv grid position (cy, cx).
  // Out-of-range output channels are computed with clamped weight reads and
  // discarded by the caller.
  fn conv4_at(cy: i32, cx: i32, oc0: u32) -> vec4<f32> {
    let wStride = KH * KW * IC4; // between consecutive output channels
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    for (var ky: u32 = 0u; ky < KH; ky = ky + 1u) {
      let iy = cy + i32(ky) - i32(p.padH);
      if (iy < 0 || iy >= i32(p.ih)) { continue; }
      for (var kx: u32 = 0u; kx < KW; kx = kx + 1u) {
        let ix = cx + i32(kx) - i32(p.padW);
        if (ix < 0 || ix >= i32(p.iw)) { continue; }
        let sBase = (u32(iy) * p.iw + u32(ix)) * IC4;
        let wBase = ((oc0 * KH + ky) * KW + kx) * IC4;
        for (var i: u32 = 0u; i < IC4; i = i + 1u) {
          let s = vec4<f32>(src.data[sBase + i]);
          acc0 = acc0 + dot(s, vec4<f32>(weight.data[wBase + i]));
          acc1 = acc1 + dot(s, vec4<f32>(weight.data[wBase + wStride + i]));
          acc2 = acc2 + dot(s, vec4<f32>(weight.data[wBase + 2u * wStride + i]));
          acc3 = acc3 + dot(s, vec4<f32>(weight.data[wBase + 3u * wStride + i]));
        }
      }
    }
    return vec4<f32>(acc0, acc1, acc2, acc3);
  }

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc0 = gid.z * 4u;
    if (ox >= p.ow || oy >= p.oh || oc0 >= p.oc) { return; }

    var acc: vec4<f32>;
    if (p.postOp == 1u) {
      let y0 = i32(oy * 2u);
      let x0 = i32(ox * 2u);
      var m = conv4_at(y0, x0, oc0);
      m = max(m, conv4_at(y0, x0 + 1, oc0));
      m = max(m, conv4_at(y0 + 1, x0, oc0));
      m = max(m, conv4_at(y0 + 1, x0 + 1, oc0));
      acc = m;
    } else if (p.postOp == 2u) {
      acc = conv4_at(i32(oy / 2u), i32(ox / 2u), oc0);
    } else {
      acc = conv4_at(i32(oy), i32(ox), oc0);
    }

    let dstBase = (oy * p.ow + ox) * p.oc + oc0;
    let n = min(4u, p.oc - oc0);
    for (var j: u32 = 0u; j < n; j = j + 1u) {
      var v = acc[j];
      if (p.biasMode == 0u) {
        v = v + f32(bias.data[oc0 + j]);
      } else if (p.biasMode == 1u) {
        v = v + f32(bias.data[dstBase + j]);
      } else {
        v = v + f32(dst.data[dstBase + j]);
      }
      if (p.activation == 1u && v < 0.0) { v = 0.0; }
      dst.data[dstBase + j] = $T$(v);
    }
  }
  )wgsl";

  // Workgroup-tiled variant of the vec4 kernel for plain (postOp == None)
  // convolutions with ic % 16 == 0: the 8x8 workgroup cooperatively stages the
  // input tile it needs (output tile + kernel halo, zero-padded at image
  // borders) into workgroup memory in 16-channel chunks, so each input pixel
  // is read from global memory once per workgroup instead of KH*KW times.
  // Weights keep coming from global memory: all 64 threads read the same
  // weight value in lockstep, which caches/broadcasts well.
  static const char* kConv2dTiledWGSL = R"wgsl(
  struct TensorF { data: array<$T$>, };
  struct TensorV { data: array<vec4<$T$>>, };

  @group(0) @binding(0) var<storage, read_write> src   : TensorV;
  @group(0) @binding(1) var<storage, read_write> weight: TensorV;
  @group(0) @binding(2) var<storage, read_write> bias  : TensorF;
  @group(0) @binding(3) var<storage, read_write> dst   : TensorF;

  struct Params { ic: u32, ih: u32, iw: u32,
                  oc: u32, oh: u32, ow: u32,
                  kh: u32, kw: u32,
                  padH: u32, padW: u32,
                  activation: u32,
                  biasMode: u32,
                  postOp: u32,
                  pad0: u32, pad1: u32, pad2: u32, };
  @group(0) @binding(4) var<uniform> p: Params;

  override IC4: u32 = 8u; // input channels / 4 (must be a multiple of C4)
  override KH: u32 = 3u;
  override KW: u32 = 3u;

  const TS: u32 = 8u; // output tile size == workgroup size
  const C4: u32 = 4u; // input vec4-channels staged per chunk
  // (TS+KH-1)*(TS+KW-1)*C4 entries; the constant size fits kernels up to 5x5
  var<workgroup> tile: array<vec4<$T$>, 576>;

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>,
          @builtin(local_invocation_index) li: u32,
          @builtin(workgroup_id) wg: vec3<u32>) {
    let oc0 = wg.z * 4u;
    let tw = TS + KW - 1u;
    let th = TS + KH - 1u;
    let tn = tw * th;
    let ty0 = i32(wg.y * TS) - i32(p.padH);
    let tx0 = i32(wg.x * TS) - i32(p.padW);
    let wStride = KH * KW * IC4;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var c0: u32 = 0u; c0 < IC4; c0 = c0 + C4) {
      // Cooperatively stage the input tile slice (zero outside the image).
      // Consecutive threads load consecutive channel vec4s of the same pixel,
      // keeping the loads coalesced.
      for (var t: u32 = li; t < tn * C4; t = t + TS * TS) {
        let pix = t / C4;
        let cc  = t % C4;
        let py = ty0 + i32(pix / tw);
        let px = tx0 + i32(pix % tw);
        var v = vec4<$T$>();
        if (py >= 0 && py < i32(p.ih) && px >= 0 && px < i32(p.iw)) {
          v = src.data[(u32(py) * p.iw + u32(px)) * IC4 + c0 + cc];
        }
        tile[t] = v;
      }
      workgroupBarrier();

      for (var ky: u32 = 0u; ky < KH; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < KW; kx = kx + 1u) {
          let tBase = ((lid.y + ky) * tw + (lid.x + kx)) * C4;
          let wBase = ((oc0 * KH + ky) * KW + kx) * IC4 + c0;
          for (var i: u32 = 0u; i < C4; i = i + 1u) {
            let s = vec4<f32>(tile[tBase + i]);
            acc0 = acc0 + dot(s, vec4<f32>(weight.data[wBase + i]));
            acc1 = acc1 + dot(s, vec4<f32>(weight.data[wBase + wStride + i]));
            acc2 = acc2 + dot(s, vec4<f32>(weight.data[wBase + 2u * wStride + i]));
            acc3 = acc3 + dot(s, vec4<f32>(weight.data[wBase + 3u * wStride + i]));
          }
        }
      }
      workgroupBarrier();
    }

    // Out-of-range threads participate in the loads/barriers above and only
    // skip the store
    let ox = gid.x;
    let oy = gid.y;
    if (ox >= p.ow || oy >= p.oh || oc0 >= p.oc) { return; }

    let acc = vec4<f32>(acc0, acc1, acc2, acc3);
    let dstBase = (oy * p.ow + ox) * p.oc + oc0;
    let n = min(4u, p.oc - oc0);
    for (var j: u32 = 0u; j < n; j = j + 1u) {
      var v = acc[j];
      if (p.biasMode == 0u) {
        v = v + f32(bias.data[oc0 + j]);
      } else if (p.biasMode == 1u) {
        v = v + f32(bias.data[dstBase + j]);
      } else {
        v = v + f32(dst.data[dstBase + j]);
      }
      if (p.activation == 1u && v < 0.0) { v = 0.0; }
      dst.data[dstBase + j] = $T$(v);
    }
  }
  )wgsl";

  // Implicit GEMM shares both input samples and weights across a matrix tile.
  // Unlike spatial-only tiling, this also reuses inputs across 32 output channels.
  static const char* kConv2dGemmWGSL = R"wgsl(
  struct TensorF { data: array<$T$>, };
  @group(0) @binding(0) var<storage, read_write> src   : TensorF;
  @group(0) @binding(1) var<storage, read_write> weight: TensorF;
  @group(0) @binding(2) var<storage, read_write> bias  : TensorF;
  @group(0) @binding(3) var<storage, read_write> dst   : TensorF;

  struct Params { ic: u32, ih: u32, iw: u32,
                  oc: u32, oh: u32, ow: u32,
                  kh: u32, kw: u32,
                  padH: u32, padW: u32,
                  activation: u32,
                  biasMode: u32,
                  postOp: u32,
                  pad0: u32, pad1: u32, pad2: u32, };
  @group(0) @binding(4) var<uniform> p: Params;

  override IC: u32 = 9u;
  override KH: u32 = 3u;
  override KW: u32 = 3u;
  // Implicit GEMM: 64 spatial positions x 32 output channels, K tiles of 16.
  var<workgroup> aTile: array<$T$, 1024>;
  var<workgroup> bTile: array<$T$, 512>;
  @compute @workgroup_size(64,1,1)
  fn main(@builtin(local_invocation_index) li:u32, @builtin(workgroup_id) wg:vec3<u32>) {
    let lane = li % 8u;
    let row = li / 8u;
    let oc0 = wg.y * 32u + lane * 4u;
    let spatialBase = (wg.x + wg.z * 65535u) * 64u;
    let pixel0 = spatialBase + row;
    let K = KH * KW * IC;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var acc4 = vec4<f32>(0.0);
    var acc5 = vec4<f32>(0.0);
    var acc6 = vec4<f32>(0.0);
    var acc7 = vec4<f32>(0.0);

    for (var base = 0u; base < K; base += 16u) {
      for (var t = li; t < 1024u; t += 64u) {
        let pixel = spatialBase + t / 16u;
        let k = base + t % 16u;
        let y = i32(pixel / p.iw) + i32(k / (KW*IC)) - i32(p.padH);
        let x = i32(pixel % p.iw) + i32((k / IC) % KW) - i32(p.padW);
        var v = $T$(0.0);
        if (pixel < p.iw*p.ih && k < K && y>=0 && y<i32(p.ih) && x>=0 && x<i32(p.iw)) {
          v = src.data[(u32(y)*p.iw+u32(x))*IC+k%IC];
        }
        aTile[t] = v;
      }
      for (var t=li; t<512u; t+=64u) {
        let oc=wg.y*32u+t/16u;
        let k=base+t%16u;
        var v=$T$(0.0);
        if(oc<p.oc && k<K) { v=weight.data[oc*K+k]; }
        bTile[(t%16u)*32u+t/16u]=v;
      }
      workgroupBarrier();
      for(var k=0u;k<16u;k++) {
        let w=k*32u+lane*4u;
        let v=vec4<f32>(f32(bTile[w]),f32(bTile[w+1u]),f32(bTile[w+2u]),f32(bTile[w+3u]));
        acc0 += f32(aTile[row*16u+k])*v;
        acc1 += f32(aTile[(row+8u)*16u+k])*v;
        acc2 += f32(aTile[(row+16u)*16u+k])*v;
        acc3 += f32(aTile[(row+24u)*16u+k])*v;
        acc4 += f32(aTile[(row+32u)*16u+k])*v;
        acc5 += f32(aTile[(row+40u)*16u+k])*v;
        acc6 += f32(aTile[(row+48u)*16u+k])*v;
        acc7 += f32(aTile[(row+56u)*16u+k])*v;

      }
      workgroupBarrier();
    }
    for(var r=0u;r<8u;r++) {
      let pixel=pixel0+r*8u;
      var accum=acc0;
      if(r==1u) { accum=acc1; }
      if(r==2u) { accum=acc2; }
      if(r==3u) { accum=acc3; }
      if(r==4u) { accum=acc4; }
      if(r==5u) { accum=acc5; }
      if(r==6u) { accum=acc6; }
      if(r==7u) { accum=acc7; }

      for(var c=0u;c<4u;c++) {
        if(pixel<p.ow*p.oh && oc0+c<p.oc) {
          let index=pixel*p.oc+oc0+c;
          var v=accum[c];
          if(p.biasMode==0u) { v+=f32(bias.data[oc0+c]); }
          else if(p.biasMode==1u) { v+=f32(bias.data[index]); }
          else { v+=f32(dst.data[index]); }
          if(p.activation==1u) { v=max(v,0.0); }
          dst.data[index]=$T$(v);
        }
      }
    }
  }
  )wgsl";

  // The channel is the innermost dimension of HWC tensors, so threads iterate
  // channel-first (gid.x = ox*C + c) to keep loads/stores coalesced
  static const char* kPoolWGSL = R"wgsl(
  struct Tensor { data: array<$T$>, };

  @group(0) @binding(0) var<storage, read_write> src : Tensor;
  @group(0) @binding(1) var<storage, read_write> dst : Tensor;
  struct Size { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, };
  @group(0) @binding(2) var<uniform> size: Size;

  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= size.ow * size.c) { return; }
    let c  = gid.x % size.c;
    let ox = gid.x / size.c;
    let oy = gid.y;
    let ix = ox * 2u;
    let iy = oy * 2u;
    let base = ((iy * size.w + ix) * size.c + c);
    let srcIdx1 = base + size.c;
    let srcIdx2 = base + size.c * size.w;
    let srcIdx3 = srcIdx2 + size.c;
    var m = src.data[base];
    m = max(m, src.data[srcIdx1]);
    m = max(m, src.data[srcIdx2]);
    m = max(m, src.data[srcIdx3]);
    dst.data[(oy * size.ow + ox) * size.c + c] = m;
  }
  )wgsl";

  static const char* kUpsampleWGSL = R"wgsl(
  struct Tensor { data: array<$T$>, };

  @group(0) @binding(0) var<storage, read_write> src : Tensor;
  @group(0) @binding(1) var<storage, read_write> dst : Tensor;
  struct Size { n: u32, c: u32, h: u32, w: u32, };
  @group(0) @binding(2) var<uniform> size: Size;

  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow = size.w * 2u;
    if (gid.x >= ow * size.c) { return; }
    let c  = gid.x % size.c;
    let ox = gid.x / size.c;
    let oy = gid.y;
    let srcIdx = ((oy / 2u) * size.w + (ox / 2u)) * size.c + c;
    dst.data[(oy * ow + ox) * size.c + c] = src.data[srcIdx];
  }
  )wgsl";

  WebGPUEngine::WebGPUEngine(WebGPUDevice* device)
    : device(device) {}

  bool WebGPUEngine::cacheUserWeights() const { return device->isExternal(); }

  WebGPUEngine::~WebGPUEngine()
  {
    if (pendingEncoder)   wgpuCommandEncoderRelease(pendingEncoder);
    for (auto& entry : pipelineCache)
    {
      wgpuComputePipelineRelease(entry.second.pipeline);
      wgpuPipelineLayoutRelease(entry.second.pipelineLayout);
      wgpuBindGroupLayoutRelease(entry.second.bindGroupLayout);
      wgpuShaderModuleRelease(entry.second.module);
    }
    if (uniformArena)     wgpuBufferRelease(uniformArena);
    if (dummyBuf)         wgpuBufferRelease(dummyBuf);
    for (auto& entry : convVariantPipelines)
      wgpuComputePipelineRelease(entry.second);
    for (int i = 0; i < 2; ++i)
    {
      if (convVec4Modules[i])  wgpuShaderModuleRelease(convVec4Modules[i]);
      if (convGemmModules[i]) wgpuShaderModuleRelease(convGemmModules[i]);
      if (convTiledModules[i]) wgpuShaderModuleRelease(convTiledModules[i]);
    }
    if (addPipeline)        wgpuComputePipelineRelease(addPipeline);
    if (addPipelineLayout)  wgpuPipelineLayoutRelease(addPipelineLayout);
    if (addBindGroupLayout) wgpuBindGroupLayoutRelease(addBindGroupLayout);
    if (addShaderModule)    wgpuShaderModuleRelease(addShaderModule);
    if (mulPipeline)        wgpuComputePipelineRelease(mulPipeline);
    if (mulPipelineLayout)  wgpuPipelineLayoutRelease(mulPipelineLayout);
    if (mulBindGroupLayout) wgpuBindGroupLayoutRelease(mulBindGroupLayout);
    if (mulShaderModule)    wgpuShaderModuleRelease(mulShaderModule);
    if (softplusPipeline)        wgpuComputePipelineRelease(softplusPipeline);
    if (softplusPipelineLayout)  wgpuPipelineLayoutRelease(softplusPipelineLayout);
    if (softplusBindGroupLayout) wgpuBindGroupLayoutRelease(softplusBindGroupLayout);
    if (softplusShaderModule)    wgpuShaderModuleRelease(softplusShaderModule);
  }

  Device* WebGPUEngine::getDevice() const
  {
    return device;
  }

  SizeAndAlignment WebGPUEngine::getBufferByteSizeAndAlignment(size_t byteSize, Storage)
  {
    // WebGPU requires storage buffer binding offsets to be 256-byte aligned
    // (minStorageBufferOffsetAlignment)
    return {byteSize, 256};
  }

  Ref<Heap> WebGPUEngine::newHeap(size_t byteSize, Storage storage)
  {
    return makeRef<WebGPUHeap>(this, byteSize, storage);
  }

  Ref<Buffer> WebGPUEngine::newBuffer(size_t byteSize, Storage)
  {
    return makeRef<WebGPUBuffer>(this, byteSize);
  }

  Ref<Buffer> WebGPUEngine::newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
  {
    return makeRef<WebGPUBuffer>(arena, byteSize, byteOffset);
  }

  Ref<Buffer> WebGPUEngine::newBuffer(void*, size_t)
  {
    throw Exception(Error::InvalidOperation,
      "creating shared buffers is not supported by the WebGPU backend");
  }

  bool WebGPUEngine::isConvSupported(Fusion fusion) const
  {
#if defined(OIDN_WEBGPU_PROFILE)
    if (profileConvMode == 1) return fusion == Fusion::None || fusion == Fusion::PoolDst;
#endif
    // A separate pool lets convolution use the tiled/GEMM kernels. The old
    // fused path was limited to the much slower direct kernel.
    return fusion == Fusion::None;
  }

  bool WebGPUEngine::isConcatConv2Supported(Fusion fusion) const
  {
    return fusion == Fusion::None;
  }

  Ref<ConcatConv2> WebGPUEngine::newConcatConv2(const ConcatConvDesc& desc)
  {
    return makeRef<InplaceConcatConv2>(this, desc);
  }

  Ref<Conv> WebGPUEngine::newConv(const ConvDesc& desc)
  {
    return makeRef<WebGPUConv>(this, desc);
  }

  Ref<Pool> WebGPUEngine::newPool(const PoolDesc& desc)
  {
    return makeRef<WebGPUPool>(this, desc);
  }

  Ref<Upsample> WebGPUEngine::newUpsample(const UpsampleDesc& desc)
  {
    return makeRef<WebGPUUpsample>(this, desc);
  }

  Ref<Autoexposure> WebGPUEngine::newAutoexposure(const ImageDesc& desc)
  {
    return makeRef<WebGPUAutoexposure>(this, desc);
  }

  Ref<InputProcess> WebGPUEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return makeRef<WebGPUInputProcess>(this, desc);
  }

  Ref<OutputProcess> WebGPUEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return makeRef<WebGPUOutputProcess>(this, desc);
  }

  Ref<ImageCopy> WebGPUEngine::newImageCopy()
  {
    return makeRef<WebGPUImageCopy>(this);
  }

  void WebGPUEngine::submitHostFunc(std::function<void()>&& f,
                                    const Ref<CancellationToken>&)
  {
    f();
  }

  void WebGPUEngine::wait()
  {
    sync();
  }

  std::string WebGPUEngine::specializeWGSL(const char* tmpl, bool f16)
  {
    std::string s = f16 ? "enable f16;\n" : "";
    s += tmpl;
    const std::string type = f16 ? "f16" : "f32";
    for (size_t pos = 0; (pos = s.find("$T$", pos)) != std::string::npos; pos += type.size())
      s.replace(pos, 3, type);
    return s;
  }

  // The scalar conv pipeline (and the shared conv bind group layout) live in
  // the generic pipeline cache, keyed per element type
  static const std::initializer_list<WGPUBufferBindingType> kConvBindings =
    {WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage,
     WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage,
     WGPUBufferBindingType_Uniform};

  WGPUComputePipeline WebGPUEngine::getConvVariantPipeline(uint32_t variant, uint32_t ic4, uint32_t kh, uint32_t kw, bool f16)
  {
    const std::array<uint32_t, 5> key = {variant, ic4, kh, kw, uint32_t(f16)};
    auto it = convVariantPipelines.find(key);
    if (it != convVariantPipelines.end())
      return it->second;

    // Reuse the scalar conv entry's pipeline layout (same binding types)
    const CachedPipeline& scalar =
      getPipeline(f16 ? "conv:f16" : "conv:f32", specializeWGSL(kConv2dWGSL, f16), kConvBindings);

    WGPUShaderModule& module = variant == 2 ? convGemmModules[f16] : variant == 1 ? convTiledModules[f16] : convVec4Modules[f16];
    if (!module)
    {
      const std::string code = specializeWGSL(variant == 2 ? kConv2dGemmWGSL : variant == 1 ? kConv2dTiledWGSL : kConv2dVec4WGSL, f16);
      OIDNWGSLSource sourceV{};
      sourceV.chain.next = nullptr;
      oidnWGSLSource(sourceV, code.c_str());

      WGPUShaderModuleDescriptor smDescV{};
      smDescV.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&sourceV);
      smDescV.label = {};
      module = wgpuDeviceCreateShaderModule(device->device, &smDescV);
    }

    WGPUConstantEntry constants[3] = {};
    constants[0].key = oidnWGPUEntryString(variant >= 2 ? "IC" : "IC4"); constants[0].value = double(ic4);
    constants[1].key = oidnWGPUEntryString("KH"); constants[1].value = double(kh);
    constants[2].key = oidnWGPUEntryString("KW"); constants[2].value = double(kw);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = scalar.pipelineLayout;
    cpDesc.compute.module = module;
    auto mainEntry = oidnWGPUEntryString("main");
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 3;
    cpDesc.compute.constants = constants;
    cpDesc.label = {};
    WGPUComputePipeline newPipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);

    convVariantPipelines.emplace(key, newPipeline);
    return newPipeline;
  }

  void WebGPUEngine::initAddPipeline()
  {
    if (addPipeline)
      return;

    static const char* kWGSL = R"wgsl(
    struct Tensor { data: array<f32>, };
    @group(0) @binding(0) var<storage, read_write> A: Tensor;
    @group(0) @binding(1) var<storage, read_write> B: Tensor;
    @group(0) @binding(2) var<storage, read_write> C: Tensor;
    @group(0) @binding(3) var<uniform> size: u32;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= size) { return; }
      C.data[i] = A.data[i] + B.data[i];
    }
    )wgsl";

    OIDNWGSLSource source{};
    source.chain.next = nullptr;
    oidnWGSLSource(source, kWGSL);

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&source);
    smDesc.label = {};

    addShaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

    WGPUBindGroupLayoutEntry entries[4] = {};
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute; entries[0].buffer.type = WGPUBufferBindingType_Storage;
    entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute; entries[1].buffer.type = WGPUBufferBindingType_Storage;
    entries[2].binding = 2; entries[2].visibility = WGPUShaderStage_Compute; entries[2].buffer.type = WGPUBufferBindingType_Storage;
    entries[3].binding = 3; entries[3].visibility = WGPUShaderStage_Compute; entries[3].buffer.type = WGPUBufferBindingType_Uniform;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 4;
    bglDesc.entries = entries;
    addBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device->device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &addBindGroupLayout;
    addPipelineLayout = wgpuDeviceCreatePipelineLayout(device->device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = addPipelineLayout;
    cpDesc.compute.module = addShaderModule;
    auto mainEntry = oidnWGPUEntryString("main");
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = {};
    addPipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  void WebGPUEngine::initMulPipeline()
  {
    if (mulPipeline)
      return;

    static const char* kWGSL = R"wgsl(
    struct Tensor { data: array<f32>, };
    @group(0) @binding(0) var<storage, read_write> A: Tensor;
    @group(0) @binding(1) var<storage, read_write> B: Tensor;
    @group(0) @binding(2) var<storage, read_write> C: Tensor;
    @group(0) @binding(3) var<uniform> size: u32;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= size) { return; }
      C.data[i] = A.data[i] * B.data[i];
    }
    )wgsl";

    OIDNWGSLSource source{};
    source.chain.next = nullptr;
    oidnWGSLSource(source, kWGSL);

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&source);
    smDesc.label = {};

    mulShaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

    WGPUBindGroupLayoutEntry entries[4] = {};
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute; entries[0].buffer.type = WGPUBufferBindingType_Storage;
    entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute; entries[1].buffer.type = WGPUBufferBindingType_Storage;
    entries[2].binding = 2; entries[2].visibility = WGPUShaderStage_Compute; entries[2].buffer.type = WGPUBufferBindingType_Storage;
    entries[3].binding = 3; entries[3].visibility = WGPUShaderStage_Compute; entries[3].buffer.type = WGPUBufferBindingType_Uniform;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 4;
    bglDesc.entries = entries;
    mulBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device->device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &mulBindGroupLayout;
    mulPipelineLayout = wgpuDeviceCreatePipelineLayout(device->device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = mulPipelineLayout;
    cpDesc.compute.module = mulShaderModule;
    auto mainEntry = oidnWGPUEntryString("main");
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = {};
    mulPipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  void WebGPUEngine::initSoftplusPipeline()
  {
    if (softplusPipeline)
      return;

    static const char* kWGSL = R"wgsl(
    struct Tensor { data: array<f32>, };
    @group(0) @binding(0) var<storage, read_write> A: Tensor;
    @group(0) @binding(1) var<storage, read_write> B: Tensor;
    @group(0) @binding(2) var<uniform> size: u32;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= size) { return; }
      let x = A.data[i];
      B.data[i] = log(1.0 + exp(x));
    }
    )wgsl";

    OIDNWGSLSource source{};
    source.chain.next = nullptr;
    oidnWGSLSource(source, kWGSL);

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&source);
    smDesc.label = {};

    softplusShaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

    WGPUBindGroupLayoutEntry entries[3] = {};
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute; entries[0].buffer.type = WGPUBufferBindingType_Storage;
    entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute; entries[1].buffer.type = WGPUBufferBindingType_Storage;
    entries[2].binding = 2; entries[2].visibility = WGPUShaderStage_Compute; entries[2].buffer.type = WGPUBufferBindingType_Uniform;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 3;
    bglDesc.entries = entries;
    softplusBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device->device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &softplusBindGroupLayout;
    softplusPipelineLayout = wgpuDeviceCreatePipelineLayout(device->device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = softplusPipelineLayout;
    cpDesc.compute.module = softplusShaderModule;
    auto mainEntry = oidnWGPUEntryString("main");
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = {};
    softplusPipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  WebGPUTensor WebGPUEngine::makeTensor(const BufferRef& buffer, WebGPUTensorType type,
                                        uint32_t n, uint32_t c, uint32_t h, uint32_t w)
  {
    Buffer* bufObj = reinterpret_cast<Buffer*>(buffer.getHandle());
    WebGPUBuffer* wb = dynamic_cast<WebGPUBuffer*>(bufObj);
    if (!wb)
      throw Exception(Error::InvalidArgument, "buffer is not a WebGPU buffer");

    size_t count = size_t(n) * c * h * w;
    if (count * sizeof(float) > wb->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer size is too small");

    return {wb->getWGPUBuffer(), wb->getWGPUBufferOffset(), n, c, h, w, type};
  }

  WGPUBuffer WebGPUEngine::getDummyBuffer()
  {
    if (!dummyBuf)
      dummyBuf = device->createBuffer(768, WGPUBufferUsage_Storage);
    return dummyBuf;
  }

  void WebGPUEngine::registerAutoexposure(const float* token, WGPUBuffer buffer)
  {
    autoexposureBuffers[token] = buffer;
  }

  void WebGPUEngine::unregisterAutoexposure(const float* token)
  {
    autoexposureBuffers.erase(token);
  }

  WGPUBuffer WebGPUEngine::getAutoexposureBuffer(const float* token) const
  {
    const auto it = autoexposureBuffers.find(token);
    return it == autoexposureBuffers.end() ? nullptr : it->second;
  }

  WebGPUEngine::UniformSlot WebGPUEngine::allocUniform(const void* data, size_t byteSize)
  {
    if (externalEncoder)
    {
      const size_t size = (byteSize + 255) & ~size_t(255);
      if (externalUniformPages.empty() || externalUniformOffset + size > uniformArenaByteSize)
      {
        externalUniformPages.push_back(device->createBuffer(uniformArenaByteSize,
          WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
        externalUniformOffset = 0;
      }
      const UniformSlot slot{externalUniformPages.back(), externalUniformOffset};
      externalUniformOffset += size;
      wgpuQueueWriteBuffer(device->queue, slot.buffer, slot.offset, data, byteSize);
      return slot;
    }
    if (!uniformArena)
      uniformArena = device->createBuffer(uniformArenaByteSize,
                                          WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);

    const size_t alignedSize = (byteSize + 255) & ~size_t(255); // binding alignment
    if (uniformArenaOffset + alignedSize > uniformArenaByteSize)
    {
      // Recorded commands have not reached the queue yet. Submit them before
      // queueWriteBuffer can overwrite slots still referenced by the encoder.
      flush();
      uniformArenaOffset = 0;
    }

    const size_t offset = uniformArenaOffset;
    uniformArenaOffset += alignedSize;

    wgpuQueueWriteBuffer(device->queue, uniformArena, offset, data, byteSize);
    return {uniformArena, offset};
  }

  WGPUCommandEncoder WebGPUEngine::getEncoder()
  {
    if (externalEncoder) return externalEncoder;
    if (device->isExternal())
      throw Exception(Error::InvalidOperation, "use oidnEncodeFilterWebGPU with an application encoder");
    if (!pendingEncoder)
      pendingEncoder = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
    return pendingEncoder;
  }

  void WebGPUEngine::flush()
  {
    if (externalEncoder) return;
    if (!pendingEncoder)
      return;
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(pendingEncoder, nullptr);
    wgpuCommandEncoderRelease(pendingEncoder);
    pendingEncoder = nullptr;
    device->submit(cmd);
    wgpuCommandBufferRelease(cmd);
  }

  void WebGPUEngine::beginExternalEncoding(WGPUCommandEncoder encoder)
  {
    if (!encoder || externalEncoder || pendingEncoder)
      throw Exception(Error::InvalidOperation, "invalid or nested external WebGPU encode");
    externalEncoder = encoder;
    externalUniformOffset = 0;
  }

  void WebGPUEngine::endExternalEncoding()
  {
    // Encoded bind groups retain their resources until execution completes.
    // Release references, never Destroy; later encodes use distinct pages.
    externalEncoder = nullptr;
    for (auto page : externalUniformPages) wgpuBufferRelease(page);
    externalUniformPages.clear();
    textureReads.clear();
    textureWrites.clear();
  }

  void WebGPUEngine::accessTexture(WebGPUTextureBuffer* texture, bool write)
  {
    if (!externalEncoder)
      throw Exception(Error::InvalidOperation, "texture images require an external WebGPU encoder");
    if (write)
    {
      if (textureWrites.insert(texture).second) texture->initializeAlpha();
    }
    else if (textureReads.insert(texture).second) texture->readTexture();
  }

  void WebGPUEngine::finishExternalTextures()
  {
    for (auto* texture : textureWrites) texture->writeTexture();
  }

  const WebGPUEngine::CachedPipeline& WebGPUEngine::getPipeline(
    const std::string& key, const std::string& code,
    std::initializer_list<WGPUBufferBindingType> bindingTypes)
  {
    std::vector<WGPUBindGroupLayoutEntry> entries(bindingTypes.size());
    uint32_t i = 0;
    for (WGPUBufferBindingType type : bindingTypes)
    {
      entries[i].binding = i;
      entries[i].visibility = WGPUShaderStage_Compute;
      entries[i].buffer.type = type;
      i++;
    }
    return getPipeline(key, code, entries);
  }

  const WebGPUEngine::CachedPipeline& WebGPUEngine::getPipeline(
    const std::string& key, const std::string& code,
    const std::vector<WGPUBindGroupLayoutEntry>& entries)
  {
    auto it = pipelineCache.find(key);
    if (it != pipelineCache.end())
      return it->second;

    CachedPipeline cp;

    OIDNWGSLSource source{};
    source.chain.next = nullptr;
    oidnWGSLSource(source, code.c_str());
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&source);
    smDesc.label = {};
    cp.module = wgpuDeviceCreateShaderModule(device->device, &smDesc);

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = entries.size();
    bglDesc.entries = entries.data();
    cp.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device->device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &cp.bindGroupLayout;
    cp.pipelineLayout = wgpuDeviceCreatePipelineLayout(device->device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = cp.pipelineLayout;
    cpDesc.compute.module = cp.module;
    auto mainEntry = oidnWGPUEntryString("main");
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = {};
    cp.pipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);

    return pipelineCache.emplace(key, cp).first->second;
  }

  void WebGPUEngine::conv2d(const WebGPUTensor& src,
                            const WebGPUTensor& weight,
                            const WebGPUTensor& bias,
                            const WebGPUTensor& dst,
                            const WebGPUConvParams& params)
  {
    const bool f16 = (src.elemByteSize == 2);
    const size_t elem = src.elemByteSize;
    const CachedPipeline& scalar =
      getPipeline(f16 ? "conv:f16" : "conv:f32", specializeWGSL(kConv2dWGSL, f16), kConvBindings);

    struct Params
    {
      uint32_t ic, ih, iw;
      uint32_t oc, oh, ow;
      uint32_t kh, kw;
      uint32_t padH, padW;
      uint32_t activation, biasMode, postOp;
      uint32_t pad0, pad1, pad2;
    } p = {src.c, src.h, src.w,
           dst.c, dst.h, dst.w,
           weight.h, weight.w,
           params.padH, params.padW,
           params.activation, params.biasMode, params.postOp,
           0, 0, 0};

    const UniformSlot paramsSlot = allocUniform(&p, sizeof(Params));

    WGPUBindGroupEntry entries[5] = {};
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].offset = src.offset; entries[0].size = roundByteSize4(size_t(src.n)*src.c*src.h*src.w*elem);
    entries[1].binding = 1; entries[1].buffer = weight.buf; entries[1].offset = weight.offset; entries[1].size = roundByteSize4(size_t(weight.n)*weight.c*weight.h*weight.w*weight.elemByteSize);
    entries[2].binding = 2; entries[2].buffer = bias.buf; entries[2].offset = bias.offset; entries[2].size = roundByteSize4(size_t(bias.n)*bias.c*bias.h*bias.w*bias.elemByteSize);
    entries[3].binding = 3; entries[3].buffer = dst.buf; entries[3].offset = dst.offset; entries[3].size = roundByteSize4(size_t(dst.n)*dst.c*dst.h*dst.w*dst.elemByteSize);
    entries[4].binding = 4; entries[4].buffer = paramsSlot.buffer; entries[4].offset = paramsSlot.offset; entries[4].size = sizeof(Params);
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = scalar.bindGroupLayout;
    bgDesc.entryCount = 5;
    bgDesc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device->device, &bgDesc);

    // The vectorized kernel requires the input channel count to be a multiple
    // of 4 (which holds for all interior UNet layers); the tiled kernel
    // additionally requires a multiple of 16 (it stages the input tile in
    // 16-channel chunks) and no fused post-op (a fused pool would quadruple
    // the staged tile)
    const bool useVec4  = (src.c % 4 == 0);
    const bool useTiled = useVec4 && params.postOp == 0 && (src.c % 16 == 0) &&
                          weight.h <= 5 && weight.w <= 5;
    // Keep the spatial kernel for narrow final outputs and tiny deep layers.
    bool useGemm = params.postOp == 0 && dst.c >= 16 && size_t(dst.w)*dst.h >= 4096;
#if defined(OIDN_WEBGPU_PROFILE)
    if (profileConvMode != 0) useGemm = false;
#endif
    WGPUComputePipeline convPipeline =
      useGemm ? getConvVariantPipeline(2, src.c, weight.h, weight.w, f16) :
      useVec4 ? getConvVariantPipeline(useTiled ? 1 : 0, src.c / 4, weight.h, weight.w, f16) : scalar.pipeline;

    WGPUCommandEncoder enc = getEncoder();
    WGPUComputePassDescriptor passDesc{};
#if defined(OIDN_WEBGPU_PROFILE)
    WGPUComputePassTimestampWrites timestamps{};
    if (profileQueries)
    {
      timestamps.querySet = profileQueries;
      timestamps.beginningOfPassWriteIndex = 2 + uint32_t(profileLabels.size()) * 2;
      timestamps.endOfPassWriteIndex = timestamps.beginningOfPassWriteIndex + 1;
      if (timestamps.endOfPassWriteIndex >= 4096)
        throw Exception(Error::InvalidOperation, "profiling query capacity exceeded");
      passDesc.timestampWrites = &timestamps;
      profileLabels.push_back(std::to_string(src.w) + "x" + std::to_string(src.h) +
        "_" + std::to_string(src.c) + "to" + std::to_string(dst.c) +
        (useGemm ? "_gemm" : useTiled ? "_tiled" : useVec4 ? "_vec4" : "_scalar") +
        (params.postOp == 1 ? "_pool" : ""));
    }
#endif
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &passDesc);
    wgpuComputePassEncoderSetPipeline(pass, convPipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    if (useGemm)
    {
      const uint32_t tiles = (dst.w*dst.h+63)/64;
      wgpuComputePassEncoderDispatchWorkgroups(pass, std::min(tiles, 65535u), (dst.c+31)/32, (tiles+65534)/65535);
    }
    else wgpuComputePassEncoderDispatchWorkgroups(pass, (dst.w+7)/8, (dst.h+7)/8, (dst.c+3)/4);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    wgpuBindGroupRelease(bg);
  }

  void WebGPUEngine::upsample2x(const WebGPUTensor& src,
                                const WebGPUTensor& dst)
  {
    const bool f16 = (src.elemByteSize == 2);
    const CachedPipeline& cp = getPipeline(
      f16 ? "upsample:f16" : "upsample:f32", specializeWGSL(kUpsampleWGSL, f16),
      {WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage, WGPUBufferBindingType_Uniform});

    struct Size
    {
      uint32_t n, c, h, w;
    } size = {src.n, src.c, src.h, src.w};

    uint32_t oh = src.h * 2;
    uint32_t ow = src.w * 2;
    const UniformSlot sizeSlot = allocUniform(&size, sizeof(Size));

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].offset = src.offset; entries[0].size = roundByteSize4(size_t(src.n)*src.c*src.h*src.w*src.elemByteSize);
    entries[1].binding = 1; entries[1].buffer = dst.buf; entries[1].offset = dst.offset; entries[1].size = roundByteSize4(size_t(dst.n)*dst.c*oh*ow*dst.elemByteSize);
    entries[2].binding = 2; entries[2].buffer = sizeSlot.buffer; entries[2].offset = sizeSlot.offset; entries[2].size = sizeof(Size);

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = cp.bindGroupLayout;
    bgDesc.entryCount = 3;
    bgDesc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device->device, &bgDesc);

    WGPUCommandEncoder enc = getEncoder();
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, cp.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (ow*src.c+255)/256, oh, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    wgpuBindGroupRelease(bg);
  }

  void WebGPUEngine::pool2x2(const WebGPUTensor& src,
                              const WebGPUTensor& dst)
  {
    const bool f16 = (src.elemByteSize == 2);
    const CachedPipeline& cp = getPipeline(
      f16 ? "pool:f16" : "pool:f32", specializeWGSL(kPoolWGSL, f16),
      {WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage, WGPUBufferBindingType_Uniform});

    struct Size
    {
      uint32_t n, c, h, w, oh, ow;
    } size = {src.n, src.c, src.h, src.w, src.h/2, src.w/2};

    uint32_t oh = src.h / 2;
    uint32_t ow = src.w / 2;
    const UniformSlot sizeSlot = allocUniform(&size, sizeof(Size));

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].offset = src.offset; entries[0].size = roundByteSize4(size_t(src.n)*src.c*src.h*src.w*src.elemByteSize);
    entries[1].binding = 1; entries[1].buffer = dst.buf; entries[1].offset = dst.offset; entries[1].size = roundByteSize4(size_t(dst.n)*dst.c*oh*ow*dst.elemByteSize);
    entries[2].binding = 2; entries[2].buffer = sizeSlot.buffer; entries[2].offset = sizeSlot.offset; entries[2].size = sizeof(Size);

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = cp.bindGroupLayout;
    bgDesc.entryCount = 3;
    bgDesc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device->device, &bgDesc);

    WGPUCommandEncoder enc = getEncoder();
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, cp.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (ow*src.c+255)/256, oh, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    wgpuBindGroupRelease(bg);
  }

  static void dispatch1D(WebGPUEngine* engine, WGPUComputePipeline pipeline,
                         WGPUBindGroupLayout layout,
                         WGPUBindGroupEntry* entries, int entryCount,
                         uint32_t size)
  {
    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = layout;
    bgDesc.entryCount = entryCount;
    bgDesc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev->device, &bgDesc);

    WGPUCommandEncoder enc = engine->getEncoder();
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (size+63)/64, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuBindGroupRelease(bg);
  }

  void WebGPUEngine::add(const WebGPUTensor& A,
                         const WebGPUTensor& B,
                         const WebGPUTensor& dst)
  {
    initAddPipeline();
    uint32_t size = A.n*A.c*A.h*A.w;
    const UniformSlot sizeSlot = allocUniform(&size, sizeof(uint32_t));
    WGPUBindGroupEntry entries[4] = {};
    entries[0].binding = 0; entries[0].buffer = A.buf; entries[0].offset = A.offset; entries[0].size = size*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = B.buf; entries[1].offset = B.offset; entries[1].size = size*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = dst.buf; entries[2].offset = dst.offset; entries[2].size = size*sizeof(float);
    entries[3].binding = 3; entries[3].buffer = sizeSlot.buffer; entries[3].offset = sizeSlot.offset; entries[3].size = sizeof(uint32_t);
    dispatch1D(this, addPipeline, addBindGroupLayout, entries, 4, size);
  }

  void WebGPUEngine::mul(const WebGPUTensor& A,
                         const WebGPUTensor& B,
                         const WebGPUTensor& dst)
  {
    initMulPipeline();
    uint32_t size = A.n*A.c*A.h*A.w;
    const UniformSlot sizeSlot = allocUniform(&size, sizeof(uint32_t));
    WGPUBindGroupEntry entries[4] = {};
    entries[0].binding = 0; entries[0].buffer = A.buf; entries[0].offset = A.offset; entries[0].size = size*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = B.buf; entries[1].offset = B.offset; entries[1].size = size*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = dst.buf; entries[2].offset = dst.offset; entries[2].size = size*sizeof(float);
    entries[3].binding = 3; entries[3].buffer = sizeSlot.buffer; entries[3].offset = sizeSlot.offset; entries[3].size = sizeof(uint32_t);
    dispatch1D(this, mulPipeline, mulBindGroupLayout, entries, 4, size);
  }

  void WebGPUEngine::softplus(const WebGPUTensor& src,
                              const WebGPUTensor& dst)
  {
    initSoftplusPipeline();
    uint32_t size = src.n*src.c*src.h*src.w;
    const UniformSlot sizeSlot = allocUniform(&size, sizeof(uint32_t));
    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].offset = src.offset; entries[0].size = size*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = dst.buf; entries[1].offset = dst.offset; entries[1].size = size*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = sizeSlot.buffer; entries[2].offset = sizeSlot.offset; entries[2].size = sizeof(uint32_t);
    dispatch1D(this, softplusPipeline, softplusBindGroupLayout, entries, 3, size);
  }

  void WebGPUEngine::sync()
  {
    flush();
    device->sync();
  }

OIDN_NAMESPACE_END
