#include "webgpu_output_process.h"
#include "webgpu_buffer.h"
#include "webgpu_device.h"
#include "core/color.h"
#include <vector>
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  WebGPUOutputProcess::WebGPUOutputProcess(WebGPUEngine* engine, const OutputProcessDesc& desc)
    : OutputProcess(desc), engine(engine) {}

  static const char* kWGSL = R"wgsl(
  struct Tensor { data: array<f32>, };
  struct Image { data: array<f32>, };
  struct Params {
    h:u32,
    w:u32,
    srcC:u32,
    dstC:u32,
    tfType:u32,
    outputScale:f32,
    normScale:f32,
    rcpNormScale:f32,
    hdr:u32,
    snormFlag:u32,
  };

  @group(0) @binding(0) var<storage, read>  src : Tensor;
  @group(0) @binding(1) var<storage, read_write> dst : Image;
  @group(0) @binding(2) var<uniform> params : Params;

  fn srgb_inverse(x:f32) -> f32 {
    let a = 12.92;
    let b = 1.055;
    let c = 1.0/2.4;
    let d = -0.055;
    let x0 = 0.04045;
    if (x <= x0) { return x / a; }
    return pow((x - d)/b, 1.0/c);
  }

  fn pu_inverse(x:f32) -> f32 {
    let A=1.41283765e+03;
    let B=1.64593172e+00;
    let C=4.31384981e-01;
    let D=-2.94139609e-03;
    let E=1.92653254e-01;
    let F=6.26026094e-03;
    let G=9.98620152e-01;
    let X0=2.23151711e-03;
    let X1=3.70974749e-01;
    if (x <= X0) { return x / A; }
    if (x <= X1) { return pow((x - D)/B, 1.0/C); }
    return exp((x - G)/E) - F;
  }

  fn tf_inverse(x: vec3<f32>) -> vec3<f32> {
    switch(params.tfType) {
      default { return x; }
      case 1u {
        return vec3<f32>(srgb_inverse(x.x), srgb_inverse(x.y), srgb_inverse(x.z));
      }
      case 2u {
        let r = vec3<f32>(pu_inverse(x.x*params.rcpNormScale), pu_inverse(x.y*params.rcpNormScale), pu_inverse(x.z*params.rcpNormScale));
        return r;
      }
      case 3u {
        return exp(x * params.rcpNormScale) - vec3<f32>(1.0);
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
    let base = ((h*params.w + w)*params.srcC);
    var value = vec3<f32>(src.data[base],
                          select(src.data[base], src.data[base+1u], params.srcC > 1u),
                          select(src.data[base], src.data[base+2u], params.srcC > 2u));
    value = sanitize(value, 0.0, 3.4028235e38);
    value = tf_inverse(value);
    if (params.dstC == 1u) {
      let m = (value.x + value.y + value.z) / 3.0;
      value = vec3<f32>(m,m,m);
    }
    if (params.snormFlag != 0u) {
      value = value * 2.0 - vec3<f32>(1.0);
      value = max(value, vec3<f32>(-1.0));
    }
    if (params.hdr == 0u) {
      value = min(value, vec3<f32>(1.0));
    }
    value = value * params.outputScale;
    let outBase = ((h*params.w + w)*params.dstC);
    dst.data[outBase] = value.x;
    if (params.dstC > 1u) { dst.data[outBase+1u] = value.y; }
    if (params.dstC > 2u) { dst.data[outBase+2u] = value.z; }
  }
  )wgsl";

  void WebGPUOutputProcess::submitKernels(const Ref<CancellationToken>&)
  {
    check();

    auto* sb = dynamic_cast<WebGPUBuffer*>(src->getBuffer());
    if (!sb)
      throw std::invalid_argument("source tensor not on WebGPU device");

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());

    const int H = srcDesc.getH();
    const int W = srcDesc.getW();
    const int C = srcDesc.getC();

    int dstC = dst->getC();
    if (dstC != 1 && dstC != 3)
      throw std::invalid_argument("unsupported image format");

    size_t outSize = size_t(H)*W*dstC*sizeof(float);
    WebGPUBuffer outBuf(engine, outSize);

    struct Params
    {
      uint32_t h,w,srcC,dstC,tfType;
      float outputScale,normScale,rcpNormScale;
      uint32_t hdr,snormFlag;
    } params = { (uint32_t)H,(uint32_t)W,(uint32_t)C,(uint32_t)dstC,(uint32_t)transferFunc->getType(),
                 transferFunc->getOutputScale(), transferFunc->normScale, transferFunc->rcpNormScale,
                 hdr?1u:0u, snorm?1u:0u };
    WGPUBuffer paramsBuf = dev->createBuffer(sizeof(Params),
                          WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                          &params);

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
    bgEntries[0].binding = 0; bgEntries[0].buffer = sb->getWGPUBuffer(); bgEntries[0].offset = src->getByteOffset(); bgEntries[0].size = size_t(C)*H*W*sizeof(float);
    bgEntries[1].binding = 1; bgEntries[1].buffer = outBuf.getWGPUBuffer(); bgEntries[1].size = outSize;
    bgEntries[2].binding = 2; bgEntries[2].buffer = paramsBuf; bgEntries[2].size = sizeof(Params);

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

    outBuf.read(0, outSize, dst->getPtr());

    wgpuBindGroupRelease(bg);
    wgpuPipelineLayoutRelease(pipelineLayout);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(shader);
    wgpuBufferRelease(paramsBuf);
  }

OIDN_NAMESPACE_END
