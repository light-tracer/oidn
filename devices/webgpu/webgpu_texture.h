// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "webgpu_buffer.h"
#include "OpenImageDenoise/oidn_webgpu.h"

OIDN_NAMESPACE_BEGIN

  // Texture-backed Image storage. The float4 buffer is a GPU conversion view,
  // reused across frames. The network continues to see standard RGB images.
  class WebGPUTextureBuffer : public WebGPUBuffer
  {
  public:
    WebGPUTextureBuffer(WebGPUEngine* engine, const OIDNWebGPUTextureDesc& desc, bool output);
    ~WebGPUTextureBuffer();
    const void* getMemoryIdentity() const override { return oidnWGPUTextureIdentity(desc.texture); }
    void readTexture();
    void initializeAlpha();
    void writeTexture();
  private:
    const WebGPUEngine::CachedPipeline& pipeline(bool write);
    void dispatch(bool write, WGPUTextureView source);
    WebGPUEngine* engine;
    OIDNWebGPUTextureDesc desc;
    WGPUTextureView view = nullptr, alpha = nullptr;
    bool output;
    std::string storageFormat;
  };

OIDN_NAMESPACE_END
