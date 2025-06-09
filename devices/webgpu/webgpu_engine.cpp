#include "webgpu_engine.h"
#include "webgpu_device.h"
#include <cmath>
#include <cstring>

OIDN_NAMESPACE_BEGIN

  static const char* kConv2dWGSL = R"wgsl(
  struct Tensor { data: array<f32>; };

  @group(0) @binding(0) var<storage, read>  src   : Tensor;
  @group(0) @binding(1) var<storage, read>  weight: Tensor;
  @group(0) @binding(2) var<storage, read>  bias  : Tensor;
  @group(0) @binding(3) var<storage, read_write> dst: Tensor;

  struct Size { n: u32, ic: u32, ih: u32, iw: u32,
                oc: u32, oh: u32, ow: u32,
                kh: u32, kw: u32 };
  @group(0) @binding(4) var<uniform> size: Size;

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;
    if (ox >= size.ow || oy >= size.oh || oc >= size.oc) { return; }

    var acc: f32 = 0.0;
    for (var ic: u32 = 0u; ic < size.ic; ic = ic + 1u) {
      for (var ky: u32 = 0u; ky < size.kh; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < size.kw; kx = kx + 1u) {
          let ix = ox + kx;
          let iy = oy + ky;
          let srcIdx = (((0u * size.ic + ic) * size.ih + iy) * size.iw + ix);
          let wIdx   = (((oc * size.ic + ic) * size.kh + ky) * size.kw + kx);
          acc = acc + src.data[srcIdx] * weight.data[wIdx];
        }
      }
    }
    acc = acc + bias.data[oc];
    if (acc < 0.0) { acc = 0.0; }
    let dstIdx = (((0u * size.oc + oc) * size.oh + oy) * size.ow + ox);
    dst.data[dstIdx] = acc;
  }
  )wgsl";

  WebGPUEngine::WebGPUEngine(WebGPUDevice* device)
    : device(device) {}

  WebGPUEngine::~WebGPUEngine()
  {
    if (pipeline)         wgpuComputePipelineRelease(pipeline);
    if (pipelineLayout)   wgpuPipelineLayoutRelease(pipelineLayout);
    if (bindGroupLayout)  wgpuBindGroupLayoutRelease(bindGroupLayout);
    if (shaderModule)     wgpuShaderModuleRelease(shaderModule);
  }

  Device* WebGPUEngine::getDevice() const
  {
    return device;
  }

  void WebGPUEngine::initPipeline()
  {
    if (pipeline)
      return;

    shaderModule = wgpuDeviceCreateShaderModule(device->device,
      &(WGPUShaderModuleDescriptor){
        .nextInChain = (const WGPUChainedStruct*)&(const WGPUShaderSourceWGSL){
          .chain = { WGPUSType_ShaderSourceWGSL },
          .code  = { kConv2dWGSL, strlen(kConv2dWGSL) }
        }
      });

    WGPUBindGroupLayoutEntry entries[5] = {};
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[2].binding = 2; entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[3].binding = 3; entries[3].visibility = WGPUShaderStage_Compute;
    entries[3].buffer.type = WGPUBufferBindingType_Storage;
    entries[4].binding = 4; entries[4].visibility = WGPUShaderStage_Compute;
    entries[4].buffer.type = WGPUBufferBindingType_Uniform;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 5;
    bglDesc.entries = entries;
    bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device->device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineLayout = wgpuDeviceCreatePipelineLayout(device->device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = pipelineLayout;
    cpDesc.compute.module = shaderModule;
    cpDesc.compute.entryPoint = "main";
    pipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  WebGPUTensor WebGPUEngine::newTensor(const float* data, WebGPUTensorType type,
                                       uint32_t n, uint32_t c, uint32_t h, uint32_t w)
  {
    size_t count = size_t(n) * c * h * w;
    size_t bytes = count * sizeof(float);
    WGPUBufferUsage usage = WGPUBufferUsage_Storage;
    if (type == WebGPUTensorType::OUTPUT)
      usage |= WGPUBufferUsage_MapRead | WGPUBufferUsage_CopySrc;
    else
      usage |= WGPUBufferUsage_CopyDst;

    WGPUBuffer buf = device->createBuffer(bytes, usage,
                                          (type == WebGPUTensorType::OUTPUT) ? nullptr : data);
    if (type == WebGPUTensorType::OUTPUT)
      outputHosts[buf] = const_cast<float*>(data);
    return {buf,0,n,c,h,w,type};
  }

  void WebGPUEngine::conv2d_eltwise(const WebGPUTensor& src,
                                    const WebGPUTensor& weight,
                                    const WebGPUTensor& bias,
                                    const WebGPUTensor& dst)
  {
    initPipeline();

    uint32_t oh = src.h - weight.h + 1;
    uint32_t ow = src.w - weight.w + 1;
    struct Size
    {
      uint32_t n, ic, ih, iw;
      uint32_t oc, oh, ow;
      uint32_t kh, kw;
    } size = {src.n, src.c, src.h, src.w, weight.n, oh, ow, weight.h, weight.w};

    WGPUBuffer sizeBuf = device->createBuffer(sizeof(Size),
                                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                                              &size);

    WGPUBindGroupEntry entries[5] = {};
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].size = src.n*src.c*src.h*src.w*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = weight.buf; entries[1].size = weight.n*weight.c*weight.h*weight.w*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = bias.buf; entries[2].size = bias.n*bias.c*bias.h*bias.w*sizeof(float);
    entries[3].binding = 3; entries[3].buffer = dst.buf; entries[3].size = dst.n*dst.c*oh*ow*sizeof(float);
    entries[4].binding = 4; entries[4].buffer = sizeBuf; entries[4].size = sizeof(Size);
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout;
    bgDesc.entryCount = 5;
    bgDesc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device->device, &bgDesc);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (ow+7)/8, (oh+7)/8, weight.n);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    if (dst.type == WebGPUTensorType::OUTPUT)
    {
      size_t outBytes = dst.n*dst.c*oh*ow*sizeof(float);
      WGPUBuffer readback = device->createBuffer(outBytes,
                          WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
      wgpuCommandEncoderCopyBufferToBuffer(enc, dst.buf, 0, readback, 0, outBytes);
      readbacks.push_back({dst.buf, readback, outputHosts[dst.buf], outBytes});
    }

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    device->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBufferRelease(sizeBuf);
    wgpuBindGroupRelease(bg);
  }

  void WebGPUEngine::sync()
  {
    device->sync();
    for (auto& rb : readbacks)
    {
      bool done = false;
      auto callback = [](WGPUBufferMapAsyncStatus status, void* userdata){
        bool* d = static_cast<bool*>(userdata); *d = (status == WGPUBufferMapAsyncStatus_Success); };
      wgpuBufferMapAsync(rb.readback, WGPUMapMode_Read, 0, rb.size,
                         (WGPUBufferMapCallbackInfo){callback, &done});
      while (!done)
        device->sync();
      std::memcpy(rb.host, wgpuBufferGetMappedRange(rb.readback,0,rb.size), rb.size);
      wgpuBufferUnmap(rb.readback);
      wgpuBufferRelease(rb.readback);
    }
    readbacks.clear();
  }

OIDN_NAMESPACE_END
