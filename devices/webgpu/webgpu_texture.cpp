// SPDX-License-Identifier: Apache-2.0
#include "webgpu_texture.h"
#include "webgpu_device.h"
#include "webgpu_compat.h"

OIDN_NAMESPACE_BEGIN

  static void validateTexture(WGPUTexture texture, uint32_t mip, uint32_t layer,
                              const OIDNWebGPUTextureDesc& desc, bool output)
  {
    if (!texture || wgpuTextureGetDimension(texture) != WGPUTextureDimension_2D ||
        wgpuTextureGetSampleCount(texture) != 1 || mip >= wgpuTextureGetMipLevelCount(texture) ||
        layer >= wgpuTextureGetDepthOrArrayLayers(texture) || !desc.width || !desc.height ||
        desc.width > ImageDesc::maxDim || desc.height > ImageDesc::maxDim)
      throw Exception(Error::InvalidArgument, "invalid WebGPU 2D texture region");
    const uint32_t width = std::max(1u, wgpuTextureGetWidth(texture) >> mip);
    const uint32_t height = std::max(1u, wgpuTextureGetHeight(texture) >> mip);
    if (!isRangeValid(desc.originX, desc.width, width) || !isRangeValid(desc.originY, desc.height, height))
      throw Exception(Error::InvalidArgument, "WebGPU texture region is out of bounds");
    const auto usage = output ? WGPUTextureUsage_StorageBinding : WGPUTextureUsage_TextureBinding;
    if (!(wgpuTextureGetUsage(texture) & usage))
      throw Exception(Error::InvalidArgument, "WebGPU texture is missing the required usage");
    if (!output)
    {
      // Float and normalized color formats; integer/depth/compressed formats
      // require a caller-owned conversion shader (e.g. LT's packed guides).
      switch (wgpuTextureGetFormat(texture))
      {
      case WGPUTextureFormat_R8Unorm: case WGPUTextureFormat_RG8Unorm:
      case WGPUTextureFormat_RGBA8Unorm: case WGPUTextureFormat_RGBA8UnormSrgb:
      case WGPUTextureFormat_BGRA8Unorm: case WGPUTextureFormat_BGRA8UnormSrgb:
      case WGPUTextureFormat_R16Float: case WGPUTextureFormat_RG16Float: case WGPUTextureFormat_RGBA16Float:
      case WGPUTextureFormat_R32Float: case WGPUTextureFormat_RG32Float: case WGPUTextureFormat_RGBA32Float:
      #if !defined(__EMSCRIPTEN__)
      case WGPUTextureFormat_RGBA16Unorm:
      #endif
        break;
      default: throw Exception(Error::InvalidArgument, "unsupported WebGPU input texture format");
      }
    }
  }

  static WGPUTextureView makeView(WGPUTexture texture, uint32_t mip, uint32_t layer)
  {
    WGPUTextureViewDescriptor desc{};
    desc.dimension = WGPUTextureViewDimension_2D;
    desc.format = wgpuTextureGetFormat(texture);
    desc.baseMipLevel = mip; desc.mipLevelCount = 1;
    desc.baseArrayLayer = layer; desc.arrayLayerCount = 1;
    desc.aspect = WGPUTextureAspect_All;
    return wgpuTextureCreateView(texture, &desc);
  }

  WebGPUTextureBuffer::WebGPUTextureBuffer(WebGPUEngine* engine, const OIDNWebGPUTextureDesc& desc, bool output)
    : WebGPUBuffer(engine, ImageDesc(Format::Float4, desc.width, desc.height).getByteSize()),
      engine(engine), desc(desc), output(output)
  {
    validateTexture(desc.texture, desc.mipLevel, desc.arrayLayer, desc, output);
    if (output)
    {
      switch (wgpuTextureGetFormat(desc.texture))
      {
      case WGPUTextureFormat_RGBA8Unorm: storageFormat = "rgba8unorm"; break;
      case WGPUTextureFormat_RGBA16Float: storageFormat = "rgba16float"; break;
      case WGPUTextureFormat_RGBA32Float: storageFormat = "rgba32float"; break;
      default: throw Exception(Error::InvalidArgument, "unsupported WebGPU storage output format");
      }
    }
    else if (desc.alphaSource)
      throw Exception(Error::InvalidArgument, "alphaSource is only valid for output images");
    if (desc.alphaSource) validateTexture(desc.alphaSource, 0, 0, desc, false);
    // Views retain the texture. No Destroy calls, including during resize.
    view = makeView(desc.texture, desc.mipLevel, desc.arrayLayer);
    if (desc.alphaSource) alpha = makeView(desc.alphaSource, 0, 0);
    pipeline(output);
    if (alpha) pipeline(false);
  }

  WebGPUTextureBuffer::~WebGPUTextureBuffer()
  {
    if (alpha) wgpuTextureViewRelease(alpha);
    if (view) wgpuTextureViewRelease(view);
  }

  const WebGPUEngine::CachedPipeline& WebGPUTextureBuffer::pipeline(bool write)
  {
    std::string code = R"wgsl(
      struct Params { width:u32, height:u32, x:u32, y:u32, alpha:u32 };
      @group(0) @binding(1) var<storage, read_write> pixels:array<vec4<f32>>;
      @group(0) @binding(2) var<uniform> p:Params;
    )wgsl";
    if (write)
      code += "@group(0) @binding(0) var image:texture_storage_2d<" + storageFormat + ",write>;\n";
    else code += "@group(0) @binding(0) var image:texture_2d<f32>;\n";
    code += R"wgsl(
      @compute @workgroup_size(8,8)
      fn main(@builtin(global_invocation_id) id:vec3<u32>) {
        if (id.x >= p.width || id.y >= p.height) { return; }
        let index = id.y*p.width+id.x;
        let position = vec2<i32>(id.xy+vec2<u32>(p.x,p.y));
    )wgsl";
    if (write)
      code += "var value=pixels[index]; if(p.alpha==0u){value.a=1.0;} textureStore(image,position,value);}";
    else code += "pixels[index]=textureLoad(image,position,0);}";
    std::vector<WGPUBindGroupLayoutEntry> entries(3);
    for (uint32_t i=0;i<3;++i) { entries[i].binding=i; entries[i].visibility=WGPUShaderStage_Compute; }
    if (write)
    {
      entries[0].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
      entries[0].storageTexture.format = wgpuTextureGetFormat(desc.texture);
      entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
    }
    else
    {
      entries[0].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
      entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    }
    entries[1].buffer.type = WGPUBufferBindingType_Storage;
    entries[2].buffer.type = WGPUBufferBindingType_Uniform;
    return engine->getPipeline(write ? "texture:write:"+storageFormat : "texture:read", code, entries);
  }

  void WebGPUTextureBuffer::dispatch(bool write, WGPUTextureView source)
  {
    auto* device = static_cast<WebGPUDevice*>(engine->getDevice());
    const auto& cp = pipeline(write);
    const uint32_t p[] = {desc.width,desc.height,desc.originX,desc.originY,alpha?1u:0u};
    const auto uniform = engine->allocUniform(p,sizeof(p));
    WGPUBindGroupEntry entries[3]{};
    entries[0].binding=0; entries[0].textureView=source;
    entries[1].binding=1; entries[1].buffer=getWGPUBuffer(); entries[1].size=getByteSize();
    entries[2].binding=2; entries[2].buffer=uniform.buffer; entries[2].offset=uniform.offset; entries[2].size=sizeof(p);
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout=cp.bindGroupLayout; bgDesc.entryCount=3; bgDesc.entries=entries;
    WGPUBindGroup bg=wgpuDeviceCreateBindGroup(device->device,&bgDesc);
    auto pass=wgpuCommandEncoderBeginComputePass(engine->getEncoder(),nullptr);
    wgpuComputePassEncoderSetPipeline(pass,cp.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass,0,bg,0,nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass,(desc.width+7)/8,(desc.height+7)/8,1);
    wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass); wgpuBindGroupRelease(bg);
  }
  void WebGPUTextureBuffer::readTexture()
  {
    if (output) throw Exception(Error::InvalidOperation,"output texture cannot be used as input");
    dispatch(false,view);
  }
  void WebGPUTextureBuffer::initializeAlpha() { if (alpha) dispatch(false,alpha); }
  void WebGPUTextureBuffer::writeTexture()
  {
    if (!output) throw Exception(Error::InvalidOperation,"input texture cannot be used as output");
    dispatch(true,view);
  }

OIDN_NAMESPACE_END
