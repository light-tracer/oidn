#include "webgpu_input_process.h"
#include "webgpu_buffer.h"
#include "webgpu_device.h"
#include "webgpu_image_util.h"
#include "core/color.h"
#include <memory>
#include <vector>
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  WebGPUInputProcess::WebGPUInputProcess(WebGPUEngine* engine, const InputProcessDesc& desc)
    : InputProcess(engine, desc), engine(engine) {}

  static const char* kWGSL = R"wgsl(
  struct Image { data: array<f32>, };
  struct Tensor { data: array<$T$>, };
  struct Params {
    h:u32,
    w:u32,
    c:u32,
    tfType:u32,
    inputScale:f32,
    normScale:f32,
    rcpNormScale:f32,
    hdr:u32,
    snormFlag:u32,
    hasAlbedo:u32,
    hasNormal:u32,
  };

  @group(0) @binding(0) var<storage, read>  inputImg : Image;
  @group(0) @binding(1) var<storage, read>  albedoImg : Image;
  @group(0) @binding(2) var<storage, read>  normalImg : Image;
  @group(0) @binding(3) var<storage, read_write> dst : Tensor;
  @group(0) @binding(4) var<uniform> params : Params;

  fn srgb_forward(y:f32) -> f32 {
    let a = 12.92;
    let b = 1.055;
    let c = 1.0/2.4;
    let d = -0.055;
    let y0 = 0.0031308;
    if (y <= y0) { return a*y; }
    return b*pow(y, c) + d;
  }

  fn pu_forward(y:f32) -> f32 {
    let A=1.41283765e+03;
    let B=1.64593172e+00;
    let C=4.31384981e-01;
    let D=-2.94139609e-03;
    let E=1.92653254e-01;
    let F=6.26026094e-03;
    let G=9.98620152e-01;
    let Y0=1.57945760e-06;
    let Y1=3.22087631e-02;
    if (y <= Y0) { return A*y; }
    if (y <= Y1) { return B*pow(y,C)+D; }
    return E*log(y+F)+G;
  }

  fn tf_forward(y: vec3<f32>) -> vec3<f32> {
    switch(params.tfType) {
      default { return y; }
      case 1u {
        return vec3<f32>(srgb_forward(y.x), srgb_forward(y.y), srgb_forward(y.z));
      }
      case 2u {
        let r = vec3<f32>(pu_forward(y.x), pu_forward(y.y), pu_forward(y.z));
        return r * params.normScale;
      }
      case 3u {
        return log(y + vec3<f32>(1.0)) * params.normScale;
      }
    }
  }

  fn sanitize(v: vec3<f32>, minV:f32, maxV:f32) -> vec3<f32> {
    var r = v;
    if (!(r.x == r.x)) { r.x = 0.0; }
    if (!(r.y == r.y)) { r.y = 0.0; }
    if (!(r.z == r.z)) { r.z = 0.0; }
    r = clamp(r, vec3<f32>(minV), vec3<f32>(maxV));
    return r;
  }

  @compute @workgroup_size(8,8,1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.y;
    let w = gid.x;
    if (h >= params.h || w >= params.w) { return; }

    let imgIdx = ((h*params.w + w)*3u);
    var value = vec3<f32>(
        inputImg.data[imgIdx],
        inputImg.data[imgIdx+1u],
        inputImg.data[imgIdx+2u]);
    value = value * params.inputScale;
    let minV = select(0.0, -1.0, params.snormFlag != 0u);
    let maxV = select(1.0, 3.4e38, params.hdr != 0u);
    value = sanitize(value, minV, maxV);
    if (params.snormFlag != 0u) {
      value = value * 0.5 + vec3<f32>(0.5);
    }
    value = tf_forward(value);

    var dstIdx = ((h*params.w + w)*params.c);
    dst.data[dstIdx] = $T$(value.x);
    if (params.c > 1u) { dst.data[dstIdx+1u] = $T$(value.y); }
    if (params.c > 2u) { dst.data[dstIdx+2u] = $T$(value.z); }
    var c = 3u;

    if (params.hasAlbedo != 0u) {
      var alb = vec3<f32>(
          albedoImg.data[imgIdx],
          albedoImg.data[imgIdx+1u],
          albedoImg.data[imgIdx+2u]);
      alb = sanitize(alb, 0.0, 1.0);
      if (c < params.c) { dst.data[dstIdx+c] = $T$(alb.x); }
      if (c+1u < params.c) { dst.data[dstIdx+c+1u] = $T$(alb.y); }
      if (c+2u < params.c) { dst.data[dstIdx+c+2u] = $T$(alb.z); }
      c = c + 3u;

      if (params.hasNormal != 0u) {
        var nrm = vec3<f32>(
            normalImg.data[imgIdx],
            normalImg.data[imgIdx+1u],
            normalImg.data[imgIdx+2u]);
        nrm = sanitize(nrm, -1.0, 1.0);
        nrm = nrm * 0.5 + vec3<f32>(0.5);
        if (c < params.c) { dst.data[dstIdx+c] = $T$(nrm.x); }
        if (c+1u < params.c) { dst.data[dstIdx+c+1u] = $T$(nrm.y); }
        if (c+2u < params.c) { dst.data[dstIdx+c+2u] = $T$(nrm.z); }
        c = c + 3u;
      }
    }

    while (c < params.c) {
      dst.data[dstIdx+c] = $T$(0.0);
      c = c + 1u;
    }
  }
  )wgsl";

  void WebGPUInputProcess::submitKernels(const Ref<CancellationToken>&)
  {
    check();

    Image* mainSrc = getMainSrc();
    if (!mainSrc || mainSrc->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");

    bool hasAlbedo = color && albedo;
    bool hasNormal = color && normal;
    if (hasAlbedo && albedo->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");
    if (hasNormal && normal->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());

    const int H = dstDesc.getH();
    const int W = dstDesc.getW();
    const int C = dstDesc.getC();

    size_t srcSize = size_t(H)*W*3*sizeof(float);
    WebGPUImageAccess srcAcc(engine, mainSrc, srcSize, false);
    std::unique_ptr<WebGPUImageAccess> albedoAcc, normalAcc;
    if (hasAlbedo)
      albedoAcc.reset(new WebGPUImageAccess(engine, albedo.get(), srcSize, false));
    if (hasNormal)
      normalAcc.reset(new WebGPUImageAccess(engine, normal.get(), srcSize, false));

    auto* dbuf = dynamic_cast<WebGPUBuffer*>(dst->getBuffer());
    if (!dbuf)
      throw std::invalid_argument("destination tensor not on WebGPU device");

    struct Params
    {
      uint32_t h,w,c,tfType;
      float inputScale,normScale,rcpNormScale;
      uint32_t hdr,snormFlag,hasAlbedo,hasNormal;
    } params = { (uint32_t)H,(uint32_t)W,(uint32_t)C,(uint32_t)transferFunc->getType(),
                 transferFunc->getInputScale(), transferFunc->normScale, transferFunc->rcpNormScale,
                 hdr ? 1u : 0u, snorm ? 1u : 0u, hasAlbedo ? 1u : 0u, hasNormal ? 1u : 0u };
    const WebGPUEngine::UniformSlot paramsSlot = engine->allocUniform(&params, sizeof(Params));

    const bool f16 = dstDesc.dataType == DataType::Float16;
    const uint32_t elem = f16 ? 2u : 4u;
    const auto& cp = engine->getPipeline(f16 ? "input_process:f16" : "input_process:f32",
      WebGPUEngine::specializeWGSL(kWGSL, f16),
      {WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_ReadOnlyStorage,
       WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Storage,
       WGPUBufferBindingType_Uniform});

    WGPUBindGroupEntry bgEntries[5] = {};
    bgEntries[0].binding = 0; bgEntries[0].buffer = srcAcc.getWGPUBuffer(); bgEntries[0].offset = srcAcc.getOffset(); bgEntries[0].size = srcSize;
    bgEntries[1].binding = 1;
    if (hasAlbedo) { bgEntries[1].buffer = albedoAcc->getWGPUBuffer(); bgEntries[1].offset = albedoAcc->getOffset(); bgEntries[1].size = srcSize; }
    else           { bgEntries[1].buffer = engine->getDummyBuffer(); bgEntries[1].size = 16; }
    bgEntries[2].binding = 2;
    if (hasNormal) { bgEntries[2].buffer = normalAcc->getWGPUBuffer(); bgEntries[2].offset = normalAcc->getOffset(); bgEntries[2].size = srcSize; }
    else           { bgEntries[2].buffer = engine->getDummyBuffer(); bgEntries[2].size = 16; }
    bgEntries[3].binding = 3; bgEntries[3].buffer = dbuf->getWGPUBuffer(); bgEntries[3].offset = dbuf->getWGPUBufferOffset() + dst->getByteOffset(); bgEntries[3].size = roundByteSize4(size_t(C)*H*W*elem);
    bgEntries[4].binding = 4; bgEntries[4].buffer = paramsSlot.buffer; bgEntries[4].offset = paramsSlot.offset; bgEntries[4].size = sizeof(Params);
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = cp.bindGroupLayout;
    bgDesc.entryCount = 5;
    bgDesc.entries = bgEntries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev->device, &bgDesc);

    WGPUCommandEncoder enc = engine->getEncoder();
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, cp.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (W+7)/8, (H+7)/8, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    wgpuBindGroupRelease(bg);
  }

OIDN_NAMESPACE_END
