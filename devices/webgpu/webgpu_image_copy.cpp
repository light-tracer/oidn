#include "webgpu_image_copy.h"
#include "webgpu_device.h"
#include "webgpu_buffer.h"
#include "webgpu_image_util.h"
#include <cstring>
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  WebGPUImageCopy::WebGPUImageCopy(WebGPUEngine* engine)
    : engine(engine) {}

  void WebGPUImageCopy::submitKernels(const Ref<CancellationToken>&)
  {
    check();

    if (src->getFormat() != Format::Float3 || dst->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());

    const int H = src->getH();
    const int W = src->getW();
    size_t byteSize = size_t(H)*W*3*sizeof(float);

    WebGPUImageAccess srcAcc(engine, src.get(), byteSize, false);
    WebGPUImageAccess dstAcc(engine, dst.get(), byteSize, true);

    static const char* kWGSL = R"wgsl(
    struct Image { data: array<f32>, };
    struct Size { h:u32, w:u32 };
    @group(0) @binding(0) var<storage, read_write> src : Image;
    @group(0) @binding(1) var<storage, read_write> dst : Image;
    @group(0) @binding(2) var<uniform> size : Size;
    @compute @workgroup_size(8,8,1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let h = gid.y;
      let w = gid.x;
      if (h >= size.h || w >= size.w) { return; }
      for (var c:u32=0u; c<3u; c=c+1u) {
        let idx = ((h*size.w + w)*3u) + c;
        dst.data[idx] = src.data[idx];
      }
    }
    )wgsl";

    struct Size { uint32_t h,w; } size = { (uint32_t)H, (uint32_t)W };
    WGPUBuffer sizeBuf = dev->createBuffer(sizeof(Size),
                        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                        &size);

    const auto& cp = engine->getPipeline(kWGSL,
      {WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage,
       WGPUBufferBindingType_Uniform});

    WGPUBindGroupEntry bgEntries[3] = {};
    bgEntries[0].binding = 0; bgEntries[0].buffer = srcAcc.getWGPUBuffer(); bgEntries[0].offset = srcAcc.getOffset(); bgEntries[0].size = byteSize;
    bgEntries[1].binding = 1; bgEntries[1].buffer = dstAcc.getWGPUBuffer(); bgEntries[1].offset = dstAcc.getOffset(); bgEntries[1].size = byteSize;
    bgEntries[2].binding = 2; bgEntries[2].buffer = sizeBuf; bgEntries[2].size = sizeof(Size);
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = cp.bindGroupLayout;
    bgDesc.entryCount = 3;
    bgDesc.entries = bgEntries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev->device, &bgDesc);

    WGPUCommandEncoder enc = engine->getEncoder();
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, cp.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (W+7)/8, (H+7)/8, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    dstAcc.finish();

    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(sizeBuf);
  }

OIDN_NAMESPACE_END
