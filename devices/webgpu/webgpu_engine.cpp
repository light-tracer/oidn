#include "webgpu_engine.h"
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
#include <cmath>
#include <cstring>

OIDN_NAMESPACE_BEGIN

  static const char* kConv2dWGSL = R"wgsl(
  struct Tensor { data: array<f32>, };

  @group(0) @binding(0) var<storage, read>  src   : Tensor;
  @group(0) @binding(1) var<storage, read>  weight: Tensor;
  @group(0) @binding(2) var<storage, read>  bias  : Tensor;
  @group(0) @binding(3) var<storage, read_write> dst: Tensor;

  struct Size { n: u32, ic: u32, ih: u32, iw: u32,
                oc: u32, oh: u32, ow: u32,
                kh: u32, kw: u32, };
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
          let srcIdx = (((0u * size.ih + iy) * size.iw + ix) * size.ic + ic);
          let wIdx   = (((oc * size.ic + ic) * size.kh + ky) * size.kw + kx);
          acc = acc + src.data[srcIdx] * weight.data[wIdx];
        }
      }
    }
    acc = acc + bias.data[oc];
    if (acc < 0.0) { acc = 0.0; }
    let dstIdx = (((0u * size.oh + oy) * size.ow + ox) * size.oc + oc);
    dst.data[dstIdx] = acc;
  }
  )wgsl";

  static const char* kPoolWGSL = R"wgsl(
  struct Tensor { data: array<f32>, };

  @group(0) @binding(0) var<storage, read>  src : Tensor;
  @group(0) @binding(1) var<storage, read_write> dst : Tensor;
  struct Size { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, };
  @group(0) @binding(2) var<uniform> size: Size;

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let c  = gid.z;
    if (ox >= size.ow || oy >= size.oh || c >= size.c) { return; }
    let ix = ox * 2u;
    let iy = oy * 2u;
    let base = (((0u * size.h + iy) * size.w + ix) * size.c + c);
    let srcIdx0 = base;
    let srcIdx1 = base + size.c;
    let srcIdx2 = base + size.c * size.w;
    let srcIdx3 = srcIdx2 + size.c;
    var m = src.data[srcIdx0];
    if (src.data[srcIdx1] > m) { m = src.data[srcIdx1]; }
    if (src.data[srcIdx2] > m) { m = src.data[srcIdx2]; }
    if (src.data[srcIdx3] > m) { m = src.data[srcIdx3]; }
    let dstIdx = (((0u * size.oh + oy) * size.ow + ox) * size.c + c);
    dst.data[dstIdx] = m;
  }
  )wgsl";

  static const char* kUpsampleWGSL = R"wgsl(
  struct Tensor { data: array<f32>, };

  @group(0) @binding(0) var<storage, read>  src : Tensor;
  @group(0) @binding(1) var<storage, read_write> dst : Tensor;
  struct Size { n: u32, c: u32, h: u32, w: u32, };
  @group(0) @binding(2) var<uniform> size: Size;

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let c  = gid.z;
    if (ox >= size.w*2u || oy >= size.h*2u || c >= size.c) { return; }
    let ix = ox / 2u;
    let iy = oy / 2u;
    let srcIdx = (((0u * size.h + iy) * size.w + ix) * size.c + c);
    let dstIdx = (((0u * (size.h*2u) + oy) * (size.w*2u) + ox) * size.c + c);
    dst.data[dstIdx] = src.data[srcIdx];
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
    if (upsamplePipeline)        wgpuComputePipelineRelease(upsamplePipeline);
    if (upsamplePipelineLayout)  wgpuPipelineLayoutRelease(upsamplePipelineLayout);
    if (upsampleBindGroupLayout) wgpuBindGroupLayoutRelease(upsampleBindGroupLayout);
    if (upsampleShaderModule)    wgpuShaderModuleRelease(upsampleShaderModule);
    if (poolPipeline)        wgpuComputePipelineRelease(poolPipeline);
    if (poolPipelineLayout)  wgpuPipelineLayoutRelease(poolPipelineLayout);
    if (poolBindGroupLayout) wgpuBindGroupLayoutRelease(poolBindGroupLayout);
    if (poolShaderModule)    wgpuShaderModuleRelease(poolShaderModule);
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
    device->sync();
  }

  void WebGPUEngine::initPipeline()
  {
    if (pipeline)
      return;

    WGPUShaderSourceWGSL source{};
    source.chain.next = nullptr;
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = { kConv2dWGSL, strlen(kConv2dWGSL) };

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&source);
    smDesc.label = { nullptr, 0 };

    shaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

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
    WGPUStringView mainEntry{ "main", WGPU_STRLEN };
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = { nullptr, 0 };
    pipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  void WebGPUEngine::initUpsamplePipeline()
  {
    if (upsamplePipeline)
      return;

    WGPUShaderSourceWGSL source{};
    source.chain.next = nullptr;
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = { kUpsampleWGSL, strlen(kUpsampleWGSL) };

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&source);
    smDesc.label = { nullptr, 0 };

    upsampleShaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

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
    upsampleBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device->device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &upsampleBindGroupLayout;
    upsamplePipelineLayout = wgpuDeviceCreatePipelineLayout(device->device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = upsamplePipelineLayout;
    cpDesc.compute.module = upsampleShaderModule;
    WGPUStringView mainEntry{ "main", WGPU_STRLEN };
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = { nullptr, 0 };
    upsamplePipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  void WebGPUEngine::initPoolPipeline()
  {
    if (poolPipeline)
      return;

    WGPUShaderSourceWGSL source{};
    source.chain.next = nullptr;
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = { kPoolWGSL, strlen(kPoolWGSL) };

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&source);
    smDesc.label = { nullptr, 0 };

    poolShaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

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
    poolBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device->device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &poolBindGroupLayout;
    poolPipelineLayout = wgpuDeviceCreatePipelineLayout(device->device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout = poolPipelineLayout;
    cpDesc.compute.module = poolShaderModule;
    WGPUStringView mainEntry{ "main", WGPU_STRLEN };
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = { nullptr, 0 };
    poolPipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  void WebGPUEngine::initAddPipeline()
  {
    if (addPipeline)
      return;

    static const char* kWGSL = R"wgsl(
    struct Tensor { data: array<f32>, };
    @group(0) @binding(0) var<storage, read>  A: Tensor;
    @group(0) @binding(1) var<storage, read>  B: Tensor;
    @group(0) @binding(2) var<storage, read_write> C: Tensor;
    @group(0) @binding(3) var<uniform> size: u32;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= size) { return; }
      C.data[i] = A.data[i] + B.data[i];
    }
    )wgsl";

    WGPUShaderSourceWGSL source{};
    source.chain.next = nullptr;
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = { kWGSL, strlen(kWGSL) };

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&source);
    smDesc.label = { nullptr, 0 };

    addShaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

    WGPUBindGroupLayoutEntry entries[4] = {};
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute; entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute; entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
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
    WGPUStringView mainEntry{ "main", WGPU_STRLEN };
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = { nullptr, 0 };
    addPipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  void WebGPUEngine::initMulPipeline()
  {
    if (mulPipeline)
      return;

    static const char* kWGSL = R"wgsl(
    struct Tensor { data: array<f32>, };
    @group(0) @binding(0) var<storage, read>  A: Tensor;
    @group(0) @binding(1) var<storage, read>  B: Tensor;
    @group(0) @binding(2) var<storage, read_write> C: Tensor;
    @group(0) @binding(3) var<uniform> size: u32;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= size) { return; }
      C.data[i] = A.data[i] * B.data[i];
    }
    )wgsl";

    WGPUShaderSourceWGSL source{};
    source.chain.next = nullptr;
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = { kWGSL, strlen(kWGSL) };

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&source);
    smDesc.label = { nullptr, 0 };

    mulShaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

    WGPUBindGroupLayoutEntry entries[4] = {};
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute; entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute; entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
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
    WGPUStringView mainEntry{ "main", WGPU_STRLEN };
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = { nullptr, 0 };
    mulPipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  void WebGPUEngine::initSoftplusPipeline()
  {
    if (softplusPipeline)
      return;

    static const char* kWGSL = R"wgsl(
    struct Tensor { data: array<f32>, };
    @group(0) @binding(0) var<storage, read>  A: Tensor;
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

    WGPUShaderSourceWGSL source{};
    source.chain.next = nullptr;
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code = { kWGSL, strlen(kWGSL) };

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&source);
    smDesc.label = { nullptr, 0 };

    softplusShaderModule = wgpuDeviceCreateShaderModule(device->device, &smDesc);

    WGPUBindGroupLayoutEntry entries[3] = {};
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute; entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
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
    WGPUStringView mainEntry{ "main", WGPU_STRLEN };
    cpDesc.compute.entryPoint = mainEntry;
    cpDesc.compute.nextInChain = nullptr;
    cpDesc.compute.constantCount = 0;
    cpDesc.compute.constants = nullptr;
    cpDesc.label = { nullptr, 0 };
    softplusPipeline = wgpuDeviceCreateComputePipeline(device->device, &cpDesc);
  }

  WebGPUTensor WebGPUEngine::makeTensor(const float* data, WebGPUTensorType type,
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

    return {wb->getWGPUBuffer(), 0, n, c, h, w, type};
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
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].offset = src.offset; entries[0].size = src.n*src.c*src.h*src.w*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = weight.buf; entries[1].offset = weight.offset; entries[1].size = weight.n*weight.c*weight.h*weight.w*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = bias.buf; entries[2].offset = bias.offset; entries[2].size = bias.n*bias.c*bias.h*bias.w*sizeof(float);
    entries[3].binding = 3; entries[3].buffer = dst.buf; entries[3].offset = dst.offset; entries[3].size = dst.n*dst.c*oh*ow*sizeof(float);
    entries[4].binding = 4; entries[4].buffer = sizeBuf; entries[4].offset = 0; entries[4].size = sizeof(Size);
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
      wgpuCommandEncoderCopyBufferToBuffer(enc, dst.buf, dst.offset, readback, 0, outBytes);
      readbacks.push_back({dst.buf, readback, outputHosts[dst.buf], outBytes});
    }

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    device->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBufferRelease(sizeBuf);
    wgpuBindGroupRelease(bg);
  }

  void WebGPUEngine::upsample2x(const WebGPUTensor& src,
                                const WebGPUTensor& dst)
  {
    initUpsamplePipeline();

    struct Size
    {
      uint32_t n, c, h, w;
    } size = {src.n, src.c, src.h, src.w};

    uint32_t oh = src.h * 2;
    uint32_t ow = src.w * 2;
    WGPUBuffer sizeBuf = device->createBuffer(sizeof(Size),
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                              &size);

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].offset = src.offset; entries[0].size = src.n*src.c*src.h*src.w*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = dst.buf; entries[1].offset = dst.offset; entries[1].size = dst.n*dst.c*oh*ow*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = sizeBuf; entries[2].offset = 0; entries[2].size = sizeof(Size);

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = upsampleBindGroupLayout;
    bgDesc.entryCount = 3;
    bgDesc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device->device, &bgDesc);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, upsamplePipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (ow+7)/8, (oh+7)/8, src.c);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    if (dst.type == WebGPUTensorType::OUTPUT)
    {
      size_t outBytes = dst.n*dst.c*oh*ow*sizeof(float);
      WGPUBuffer readback = device->createBuffer(outBytes,
                          WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
      wgpuCommandEncoderCopyBufferToBuffer(enc, dst.buf, dst.offset, readback, 0, outBytes);
      readbacks.push_back({dst.buf, readback, outputHosts[dst.buf], outBytes});
    }

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    device->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBufferRelease(sizeBuf);
    wgpuBindGroupRelease(bg);
  }

  void WebGPUEngine::pool2x2(const WebGPUTensor& src,
                              const WebGPUTensor& dst)
  {
    initPoolPipeline();

    struct Size
    {
      uint32_t n, c, h, w, oh, ow;
    } size = {src.n, src.c, src.h, src.w, src.h/2, src.w/2};

    uint32_t oh = src.h / 2;
    uint32_t ow = src.w / 2;
    WGPUBuffer sizeBuf = device->createBuffer(sizeof(Size),
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                              &size);

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].size = src.n*src.c*src.h*src.w*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = dst.buf; entries[1].size = dst.n*dst.c*oh*ow*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = sizeBuf; entries[2].size = sizeof(Size);

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = poolBindGroupLayout;
    bgDesc.entryCount = 3;
    bgDesc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device->device, &bgDesc);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, poolPipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (ow+7)/8, (oh+7)/8, src.c);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    if (dst.type == WebGPUTensorType::OUTPUT)
    {
      size_t outBytes = dst.n*dst.c*oh*ow*sizeof(float);
      WGPUBuffer readback = device->createBuffer(outBytes,
                          WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
      wgpuCommandEncoderCopyBufferToBuffer(enc, dst.buf, dst.offset, readback, 0, outBytes);
      readbacks.push_back({dst.buf, readback, outputHosts[dst.buf], outBytes});
    }

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    device->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBufferRelease(sizeBuf);
    wgpuBindGroupRelease(bg);
  }

  static void dispatch1D(WebGPUDevice* dev, WGPUComputePipeline pipeline,
                         WGPUBindGroupLayout layout,
                         WGPUBindGroupEntry* entries, int entryCount,
                         uint32_t size)
  {
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = layout;
    bgDesc.entryCount = entryCount;
    bgDesc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev->device, &bgDesc);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev->device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (size+63)/64, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    dev->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBindGroupRelease(bg);
  }

  void WebGPUEngine::add(const WebGPUTensor& A,
                         const WebGPUTensor& B,
                         const WebGPUTensor& dst)
  {
    initAddPipeline();
    uint32_t size = A.n*A.c*A.h*A.w;
    WGPUBuffer sizeBuf = device->createBuffer(sizeof(uint32_t),
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                              &size);
    WGPUBindGroupEntry entries[4] = {};
    entries[0].binding = 0; entries[0].buffer = A.buf; entries[0].offset = A.offset; entries[0].size = size*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = B.buf; entries[1].offset = B.offset; entries[1].size = size*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = dst.buf; entries[2].offset = dst.offset; entries[2].size = size*sizeof(float);
    entries[3].binding = 3; entries[3].buffer = sizeBuf; entries[3].offset = 0; entries[3].size = sizeof(uint32_t);
    dispatch1D(device, addPipeline, addBindGroupLayout, entries, 4, size);
    if (dst.type == WebGPUTensorType::OUTPUT)
    {
      WGPUBuffer readback = device->createBuffer(size*sizeof(float),
                          WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
      WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
      wgpuCommandEncoderCopyBufferToBuffer(enc, dst.buf, dst.offset, readback, 0, size*sizeof(float));
      WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
      device->submit(cmd);
      wgpuCommandBufferRelease(cmd);
      wgpuCommandEncoderRelease(enc);
      readbacks.push_back({dst.buf, readback, outputHosts[dst.buf], size*sizeof(float)});
    }
    wgpuBufferRelease(sizeBuf);
  }

  void WebGPUEngine::mul(const WebGPUTensor& A,
                         const WebGPUTensor& B,
                         const WebGPUTensor& dst)
  {
    initMulPipeline();
    uint32_t size = A.n*A.c*A.h*A.w;
    WGPUBuffer sizeBuf = device->createBuffer(sizeof(uint32_t),
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                              &size);
    WGPUBindGroupEntry entries[4] = {};
    entries[0].binding = 0; entries[0].buffer = A.buf; entries[0].offset = A.offset; entries[0].size = size*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = B.buf; entries[1].offset = B.offset; entries[1].size = size*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = dst.buf; entries[2].offset = dst.offset; entries[2].size = size*sizeof(float);
    entries[3].binding = 3; entries[3].buffer = sizeBuf; entries[3].offset = 0; entries[3].size = sizeof(uint32_t);
    dispatch1D(device, mulPipeline, mulBindGroupLayout, entries, 4, size);
    if (dst.type == WebGPUTensorType::OUTPUT)
    {
      WGPUBuffer readback = device->createBuffer(size*sizeof(float),
                          WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
      WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
      wgpuCommandEncoderCopyBufferToBuffer(enc, dst.buf, dst.offset, readback, 0, size*sizeof(float));
      WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
      device->submit(cmd);
      wgpuCommandBufferRelease(cmd);
      wgpuCommandEncoderRelease(enc);
      readbacks.push_back({dst.buf, readback, outputHosts[dst.buf], size*sizeof(float)});
    }
    wgpuBufferRelease(sizeBuf);
  }

  void WebGPUEngine::softplus(const WebGPUTensor& src,
                              const WebGPUTensor& dst)
  {
    initSoftplusPipeline();
    uint32_t size = src.n*src.c*src.h*src.w;
    WGPUBuffer sizeBuf = device->createBuffer(sizeof(uint32_t),
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                              &size);
    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = src.buf; entries[0].offset = src.offset; entries[0].size = size*sizeof(float);
    entries[1].binding = 1; entries[1].buffer = dst.buf; entries[1].offset = dst.offset; entries[1].size = size*sizeof(float);
    entries[2].binding = 2; entries[2].buffer = sizeBuf; entries[2].offset = 0; entries[2].size = sizeof(uint32_t);
    dispatch1D(device, softplusPipeline, softplusBindGroupLayout, entries, 3, size);
    if (dst.type == WebGPUTensorType::OUTPUT)
    {
      WGPUBuffer readback = device->createBuffer(size*sizeof(float),
                          WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
      WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
      wgpuCommandEncoderCopyBufferToBuffer(enc, dst.buf, dst.offset, readback, 0, size*sizeof(float));
      WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
      device->submit(cmd);
      wgpuCommandBufferRelease(cmd);
      wgpuCommandEncoderRelease(enc);
      readbacks.push_back({dst.buf, readback, outputHosts[dst.buf], size*sizeof(float)});
    }
    wgpuBufferRelease(sizeBuf);
  }

  void WebGPUEngine::sync()
  {
    device->sync();
    for (auto& rb : readbacks)
    {
      bool done = false;
      auto callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*)
      {
        bool* d = static_cast<bool*>(userdata1);
        *d = (status == WGPUMapAsyncStatus_Success);
      };
      WGPUBufferMapCallbackInfo cbInfo{};
      cbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
      cbInfo.callback = callback;
      cbInfo.userdata1 = &done;
      wgpuBufferMapAsync(rb.readback, WGPUMapMode_Read, 0, rb.size, cbInfo);
      while (!done)
        device->sync();
      std::memcpy(rb.host, wgpuBufferGetMappedRange(rb.readback,0,rb.size), rb.size);
      wgpuBufferUnmap(rb.readback);
      wgpuBufferRelease(rb.readback);
    }
    readbacks.clear();
  }

OIDN_NAMESPACE_END
