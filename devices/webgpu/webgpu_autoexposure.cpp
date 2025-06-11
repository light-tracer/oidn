#include "webgpu_autoexposure.h"
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

    WGPUBuffer srcBuf = dev->createBuffer(byteSize,
                    WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                    src->getPtr());
    WebGPUBuffer dstBuf(engine, sizeof(float));

    static const char* kWGSL = R"wgsl(
    struct Image { data: array<f32>, };
    struct Size { h:u32, w:u32 };
    @group(0) @binding(0) var<storage, read>  src : Image;
    @group(0) @binding(1) var<storage, read_write> dst : array<f32>;
    @group(0) @binding(2) var<uniform> size : Size;
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      var sum: f32 = 0.0;
      for (var h:u32=0u; h<size.h; h=h+1u) {
        for (var w:u32=0u; w<size.w; w=w+1u) {
          let idx = (h*size.w + w)*3u;
          let L = 0.2126*src.data[idx] + 0.7152*src.data[idx+1u] + 0.0722*src.data[idx+2u];
          sum = sum + L;
        }
      }
      let Lavg = sum / f32(size.h * size.w);
      var exposure: f32;
      if (Lavg > ${eps}) {
        exposure = ${key} / Lavg;
      } else {
        exposure = 1.0;
      }
      dst[0] = exposure;
    }
    )wgsl";

    std::string shaderSrc = kWGSL;
    // Replace constants
    auto replaceAll=[&](std::string& s,const char* from,const std::string& to){size_t pos=0;while((pos=s.find(from,pos))!=std::string::npos){s.replace(pos,strlen(from),to);pos+=to.size();}};
    replaceAll(shaderSrc,"${eps}",std::to_string(eps));
    replaceAll(shaderSrc,"${key}",std::to_string(key));

    WGPUShaderSourceWGSL source{};
    source.chain.next = nullptr;
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = { shaderSrc.c_str(), shaderSrc.size() };
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

    struct Size { uint32_t h,w; } size = { (uint32_t)H, (uint32_t)W };
    WGPUBuffer sizeBuf = dev->createBuffer(sizeof(Size),
                        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                        &size);

    WGPUBindGroupEntry bgEntries[3] = {};
    bgEntries[0].binding = 0; bgEntries[0].buffer = srcBuf; bgEntries[0].size = byteSize;
    bgEntries[1].binding = 1; bgEntries[1].buffer = dstBuf.getWGPUBuffer(); bgEntries[1].size = sizeof(float);
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
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1,1,1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    dev->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    dstBuf.read(0, sizeof(float), getDstPtr());

    wgpuBindGroupRelease(bg);
    wgpuPipelineLayoutRelease(pipelineLayout);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(shader);
    wgpuBufferRelease(sizeBuf);
    wgpuBufferRelease(srcBuf);
  }

OIDN_NAMESPACE_END
