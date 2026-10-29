#include "webgpu_autoexposure.h"
#include "webgpu_image_util.h"
#include "webgpu_buffer.h"
#include "webgpu_device.h"

OIDN_NAMESPACE_BEGIN

  WebGPUAutoexposure::WebGPUAutoexposure(WebGPUEngine* engine, const ImageDesc& srcDesc)
    : Autoexposure(srcDesc), engine(engine)
  {
    binsBuffer = engine->newBuffer(size_t(numBins) * sizeof(float), Storage::Device);
    scaleBuffer = engine->newBuffer(sizeof(float), Storage::Device);
    engine->registerAutoexposure(&hostResult,
      static_cast<WebGPUBuffer*>(scaleBuffer.get())->getWGPUBuffer());
  }

  WebGPUAutoexposure::~WebGPUAutoexposure()
  {
    engine->unregisterAutoexposure(&hostResult);
  }

  // Match CPU bin boundaries and luminance, then reduce the bins on the GPU.
  static const char* kBinsWGSL = R"wgsl(
    struct ImageParams { offset:u32, hStride:u32, wStride:u32, channels:u32 };
    struct Params { image:ImageParams, h:u32, w:u32, numBinsH:u32, numBinsW:u32 };
    @group(0) @binding(0) var<storage, read_write> src : array<f32>;
    @group(0) @binding(1) var<storage, read_write> bins : array<f32>;
    @group(0) @binding(2) var<uniform> p : Params;
    @compute @workgroup_size(8,8,1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.y;
      let j = gid.x;
      if (i >= p.numBinsH || j >= p.numBinsW) { return; }
      let beginH = (i * p.h) / p.numBinsH;
      let beginW = (j * p.w) / p.numBinsW;
      let endH = ((i + 1u) * p.h) / p.numBinsH;
      let endW = ((j + 1u) * p.w) / p.numBinsW;
      var sum = 0.0;
      for (var h = beginH; h < endH; h++) {
        for (var w = beginW; w < endW; w++) {
          let idx = p.image.offset + h*p.image.hStride + w*p.image.wStride;
          var r = src[idx];
          var g = src[idx+min(1u, p.image.channels-1u)];
          var b = src[idx+min(2u, p.image.channels-1u)];
          if (!(r == r)) { r = 0.0; }
          if (!(g == g)) { g = 0.0; }
          if (!(b == b)) { b = 0.0; }
          r = max(r, 0.0); g = max(g, 0.0); b = max(b, 0.0);
          sum += 0.212671*r + 0.715160*g + 0.072169*b;
        }
      }
      bins[i*p.numBinsW+j] = sum / f32((endH-beginH)*(endW-beginW));
    }
  )wgsl";

  static const char* kReduceWGSL = R"wgsl(
    struct Params { numBins:u32, eps:f32, key:f32, pad:u32 };
    @group(0) @binding(0) var<storage, read_write> bins : array<f32>;
    @group(0) @binding(1) var<storage, read_write> scale : array<f32>;
    @group(0) @binding(2) var<uniform> p : Params;
    var<workgroup> sums : array<f32, 256>;
    var<workgroup> counts : array<u32, 256>;
    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_index) lane:u32) {
      var sum = 0.0;
      var count = 0u;
      for (var i = lane; i < p.numBins; i += 256u) {
        if (bins[i] > p.eps) { sum += log2(bins[i]); count++; }
      }
      sums[lane] = sum; counts[lane] = count;
      workgroupBarrier();
      for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
          sums[lane] += sums[lane+stride];
          counts[lane] += counts[lane+stride];
        }
        workgroupBarrier();
      }
      if (lane == 0u) {
        scale[0] = 1.0;
        if (counts[0] > 0u) { scale[0] = p.key / exp2(sums[0]/f32(counts[0])); }
      }
    }
  )wgsl";

  void WebGPUAutoexposure::submitKernels(const Ref<CancellationToken>&)
  {
    if (!src || !dst)
      throw std::logic_error("autoexposure source/destination not set");

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    WebGPUImageAccess srcAcc(engine, src.get(), false);
    auto* bins = static_cast<WebGPUBuffer*>(binsBuffer.get());
    auto* scale = static_cast<WebGPUBuffer*>(scaleBuffer.get());
    const auto bindings = {WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage,
                           WGPUBufferBindingType_Uniform};
    const auto& binsPipeline = engine->getPipeline("autoexposure:bins", kBinsWGSL, bindings);
    const auto& reducePipeline = engine->getPipeline("autoexposure:reduce", kReduceWGSL, bindings);

    auto dispatch = [&](const WebGPUEngine::CachedPipeline& cp, WGPUBindGroupEntry* entries,
                        uint32_t x, uint32_t y)
    {
      WGPUBindGroupDescriptor desc{};
      desc.layout = cp.bindGroupLayout; desc.entryCount = 3; desc.entries = entries;
      WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->device, &desc);
      WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(engine->getEncoder(), nullptr);
      wgpuComputePassEncoderSetPipeline(pass, cp.pipeline);
      wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
      wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
      wgpuBindGroupRelease(group);
    };

    struct BinsParams { WebGPUImageParams image; uint32_t h,w,numBinsH,numBinsW; };
    const BinsParams binsParams{srcAcc.getParams(), uint32_t(srcDesc.getH()), uint32_t(srcDesc.getW()),
                                uint32_t(numBinsH), uint32_t(numBinsW)};
    auto slot = engine->allocUniform(&binsParams, sizeof(binsParams));
    WGPUBindGroupEntry entries[3]{};
    entries[0].binding = 0; entries[0].buffer = srcAcc.getWGPUBuffer();
    entries[0].offset = srcAcc.getOffset(); entries[0].size = srcAcc.getByteSize();
    entries[1].binding = 1; entries[1].buffer = bins->getWGPUBuffer(); entries[1].size = binsBuffer->getByteSize();
    entries[2].binding = 2; entries[2].buffer = slot.buffer; entries[2].offset = slot.offset; entries[2].size = sizeof(binsParams);
    dispatch(binsPipeline, entries, (numBinsW+7)/8, (numBinsH+7)/8);

    struct ReduceParams { uint32_t numBins; float eps, key; uint32_t pad; };
    const ReduceParams reduceParams{uint32_t(numBins), eps, key, 0};
    slot = engine->allocUniform(&reduceParams, sizeof(reduceParams));
    entries[0].buffer = bins->getWGPUBuffer(); entries[0].offset = 0; entries[0].size = binsBuffer->getByteSize();
    entries[1].buffer = scale->getWGPUBuffer(); entries[1].size = sizeof(float);
    entries[2].buffer = slot.buffer; entries[2].offset = slot.offset; entries[2].size = sizeof(reduceParams);
    dispatch(reducePipeline, entries, 1, 1);
  }

OIDN_NAMESPACE_END
