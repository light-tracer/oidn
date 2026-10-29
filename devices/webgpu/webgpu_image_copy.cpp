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

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());

    const int H = src->getH();
    const int W = src->getW();
    WebGPUImageAccess srcAcc(engine, src.get(), false);
    WebGPUImageAccess dstAcc(engine, dst.get(), true);

    static const char* kWGSL = R"wgsl(
    struct Image { data: array<f32>, };
    struct ImageParams { offset:u32, hStride:u32, wStride:u32, channels:u32 };
    struct Size { src:ImageParams, dst:ImageParams, h:u32, w:u32 };
    @group(0) @binding(0) var<storage, read_write> src : Image;
    @group(0) @binding(1) var<storage, read_write> dst : Image;
    @group(0) @binding(2) var<uniform> size : Size;
    @compute @workgroup_size(8,8,1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let h = gid.y;
      let w = gid.x;
      if (h >= size.h || w >= size.w) { return; }
      let srcIdx = size.src.offset + h*size.src.hStride + w*size.src.wStride;
      let dstIdx = size.dst.offset + h*size.dst.hStride + w*size.dst.wStride;
      for (var c:u32=0u; c<size.dst.channels; c=c+1u) {
        dst.data[dstIdx+c] = src.data[srcIdx+min(c,size.src.channels-1u)];
      }
    }
    )wgsl";

    struct Size { WebGPUImageParams src,dst; uint32_t h,w; };
    Size size = { srcAcc.getParams(), dstAcc.getParams(), (uint32_t)H, (uint32_t)W };
    const WebGPUEngine::UniformSlot sizeSlot = engine->allocUniform(&size, sizeof(Size));

    const auto& cp = engine->getPipeline("image_copy", kWGSL,
      {WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage,
       WGPUBufferBindingType_Uniform});

    WGPUBindGroupEntry bgEntries[3] = {};
    bgEntries[0].binding = 0; bgEntries[0].buffer = srcAcc.getWGPUBuffer(); bgEntries[0].offset = srcAcc.getOffset(); bgEntries[0].size = srcAcc.getByteSize();
    bgEntries[1].binding = 1; bgEntries[1].buffer = dstAcc.getWGPUBuffer(); bgEntries[1].offset = dstAcc.getOffset(); bgEntries[1].size = dstAcc.getByteSize();
    bgEntries[2].binding = 2; bgEntries[2].buffer = sizeSlot.buffer; bgEntries[2].offset = sizeSlot.offset; bgEntries[2].size = sizeof(Size);
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
  }

OIDN_NAMESPACE_END
