#include "webgpu_autoexposure.h"
#include "webgpu_image_util.h"
#include "core/color.h"
#include <vector>
#include <cmath>
#include <string>
#include "webgpu_device.h"
#include "webgpu_buffer.h"
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  WebGPUAutoexposure::WebGPUAutoexposure(WebGPUEngine* engine, const ImageDesc& srcDesc)
    : Autoexposure(srcDesc), engine(engine) {}

  // Matches the CPU implementation: downsample the image into bins of about
  // maxBinSize x maxBinSize pixels (one invocation per bin, computing the
  // average luminance), then compute the average log luminance of the bins on
  // the host and derive the exposure from it.
  static const char* kWGSL = R"wgsl(
  struct Image { data: array<f32>, };
  struct Params { h:u32, w:u32, numBinsH:u32, numBinsW:u32 };
  @group(0) @binding(0) var<storage, read>  src : Image;
  @group(0) @binding(1) var<storage, read_write> bins : array<f32>;
  @group(0) @binding(2) var<uniform> p : Params;

  @compute @workgroup_size(8,8,1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y; // bin row
    let j = gid.x; // bin column
    if (i >= p.numBinsH || j >= p.numBinsW) { return; }

    let beginH = (i * p.h) / p.numBinsH;
    let beginW = (j * p.w) / p.numBinsW;
    let endH = ((i + 1u) * p.h) / p.numBinsH;
    let endW = ((j + 1u) * p.w) / p.numBinsW;

    var sum: f32 = 0.0;
    for (var h = beginH; h < endH; h = h + 1u) {
      for (var w = beginW; w < endW; w = w + 1u) {
        let idx = (h * p.w + w) * 3u;
        var r = src.data[idx];
        var g = src.data[idx + 1u];
        var b = src.data[idx + 2u];
        if (!(r == r)) { r = 0.0; }
        if (!(g == g)) { g = 0.0; }
        if (!(b == b)) { b = 0.0; }
        r = max(r, 0.0);
        g = max(g, 0.0);
        b = max(b, 0.0);
        sum = sum + 0.212671*r + 0.715160*g + 0.072169*b;
      }
    }
    bins[i * p.numBinsW + j] = sum / f32((endH - beginH) * (endW - beginW));
  }
  )wgsl";

  void WebGPUAutoexposure::submitKernels(const Ref<CancellationToken>&)
  {
    if (!src || !dst)
      throw std::logic_error("autoexposure source/destination not set");
    if (src->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());

    const int H = srcDesc.getH();
    const int W = srcDesc.getW();
    size_t byteSize = size_t(H)*W*3*sizeof(float);

    WebGPUImageAccess srcAcc(engine, src.get(), byteSize, false);
    WebGPUBuffer binsBuf(engine, size_t(numBins)*sizeof(float));

    const auto& cp = engine->getPipeline(kWGSL,
      {WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Storage,
       WGPUBufferBindingType_Uniform});

    struct Params { uint32_t h,w,numBinsH,numBinsW; }
      params = { (uint32_t)H, (uint32_t)W, (uint32_t)numBinsH, (uint32_t)numBinsW };
    WGPUBuffer paramsBuf = dev->createBuffer(sizeof(Params),
                        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                        &params);

    WGPUBindGroupEntry bgEntries[3] = {};
    bgEntries[0].binding = 0; bgEntries[0].buffer = srcAcc.getWGPUBuffer(); bgEntries[0].offset = srcAcc.getOffset(); bgEntries[0].size = byteSize;
    bgEntries[1].binding = 1; bgEntries[1].buffer = binsBuf.getWGPUBuffer(); bgEntries[1].size = size_t(numBins)*sizeof(float);
    bgEntries[2].binding = 2; bgEntries[2].buffer = paramsBuf; bgEntries[2].size = sizeof(Params);
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = cp.bindGroupLayout;
    bgDesc.entryCount = 3;
    bgDesc.entries = bgEntries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev->device, &bgDesc);

    WGPUCommandEncoder enc = engine->getEncoder();
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, cp.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (numBinsW+7)/8, (numBinsH+7)/8, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    // Reduce the bins on the host (matches the CPU backend numerically);
    // the blocking read flushes the pending encoder first
    std::vector<float> bins(numBins);
    binsBuf.read(0, size_t(numBins)*sizeof(float), bins.data());

    double logSum = 0;
    int count = 0;
    for (float L : bins)
    {
      if (L > eps)
      {
        logSum += math::log2(L);
        count++;
      }
    }
    *getDstPtr() = (count > 0) ? (key / math::exp2(float(logSum / count))) : 1.f;

    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(paramsBuf);
  }

OIDN_NAMESPACE_END
