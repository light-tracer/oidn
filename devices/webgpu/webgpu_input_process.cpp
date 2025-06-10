#include "webgpu_input_process.h"
#include "webgpu_buffer.h"
#include "webgpu_device.h"
#include "core/color.h"
#include <vector>
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  WebGPUInputProcess::WebGPUInputProcess(WebGPUEngine* engine, const InputProcessDesc& desc)
    : InputProcess(engine, desc), engine(engine) {}

  static const char* kWGSL = R"wgsl(
  struct Image { data: array<f32>; };
  struct Tensor { data: array<f32>; };
  struct Size { h:u32, w:u32, c:u32 }; 

  @group(0) @binding(0) var<storage, read>  src : Image;
  @group(0) @binding(1) var<storage, read_write> dst : Tensor;
  @group(0) @binding(2) var<uniform> size : Size;

  @compute @workgroup_size(8,8,1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.y;
    let w = gid.x;
    if (h >= size.h || w >= size.w) { return; }
    for (var i:u32 = 0u; i < size.c; i = i + 1u) {
      let srcIdx = ((h*size.w + w)*3u) + i;
      let dstIdx = ((h*size.w + w)*size.c) + i;
      dst.data[dstIdx] = src.data[srcIdx];
    }
  }
  )wgsl";

  void WebGPUInputProcess::submitKernels(const Ref<CancellationToken>&)
  {
    check();

    if (!color || albedo || normal)
      throw std::logic_error("unsupported input process configuration");
    if (color->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());

    const int H = dstDesc.getH();
    const int W = dstDesc.getW();
    const int C = dstDesc.getC();

    size_t srcSize = size_t(H)*W*3*sizeof(float);
    WGPUBuffer srcBuf = dev->createBuffer(srcSize,
                    WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                    color->getPtr());

    auto* dbuf = dynamic_cast<WebGPUBuffer*>(dst->getBuffer());
    if (!dbuf)
      throw std::invalid_argument("destination tensor not on WebGPU device");

    struct Size { uint32_t h,w,c; } size = { (uint32_t)H, (uint32_t)W, (uint32_t)C };
    WGPUBuffer sizeBuf = dev->createBuffer(sizeof(Size),
                        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                        &size);

    WGPUShaderSourceWGSL source{};
    source.chain.next = nullptr;
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = { kWGSL, strlen(kWGSL) };
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&source);
    smDesc.label = { nullptr, 0 };
    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(dev->device, &smDesc);

    WGPUBindGroupLayoutEntry entries[3] = {};
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.type = WGPUBufferBindingType_Storage;
    entries[2].binding = 2; entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].buffer.type = WGPUBufferBindingType_Uniform;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 3;
    bglDesc.entries = entries;
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(dev->device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(dev->device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = pipelineLayout;
    cpDesc.compute.module = shader;
    WGPUStringView mainEntry{ "main", WGPU_STRLEN };
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = { nullptr, 0 };
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(dev->device, &cpDesc);

    WGPUBindGroupEntry bgEntries[3] = {};
    bgEntries[0].binding = 0; bgEntries[0].buffer = srcBuf; bgEntries[0].size = srcSize;
    bgEntries[1].binding = 1; bgEntries[1].buffer = dbuf->getWGPUBuffer(); bgEntries[1].offset = dst->getByteOffset(); bgEntries[1].size = size_t(C)*H*W*sizeof(float);
    bgEntries[2].binding = 2; bgEntries[2].buffer = sizeBuf; bgEntries[2].size = sizeof(Size);
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = bgl;
    bgDesc.entryCount = 3;
    bgDesc.entries = bgEntries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev->device, &bgDesc);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev->device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (W+7)/8, (H+7)/8, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    dev->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    wgpuBindGroupRelease(bg);
    wgpuPipelineLayoutRelease(pipelineLayout);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(shader);
    wgpuBufferRelease(sizeBuf);
    wgpuBufferRelease(srcBuf);
  }

OIDN_NAMESPACE_END
