#pragma once
#include "core/engine.h"
#include "OpenImageDenoise/webgpu.h"
#include "OpenImageDenoise/oidn.hpp"
#include <array>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

OIDN_NAMESPACE_BEGIN

  class WebGPUDevice;
  class WebGPUInputProcess;
  class WebGPUOutputProcess;
  class WebGPUImageCopy;
  class WebGPUAutoexposure;
  class WebGPUTextureBuffer;

  // WebGPU requires buffer sizes, copy/map sizes and storage binding sizes to
  // be multiples of 4 bytes. f16 tensor regions can be 2-aligned only (e.g. a
  // 3-channel bias), so such sizes must be rounded up.
  inline size_t roundByteSize4(size_t byteSize)
  {
    return (byteSize + 3) & ~size_t(3);
  }

  class WebGPUEngine : public Engine
  {
  public:
    explicit WebGPUEngine(WebGPUDevice* device);
    ~WebGPUEngine();

    Device* getDevice() const override;
    bool cacheUserWeights() const override;

    // Heap / Buffer
    SizeAndAlignment getBufferByteSizeAndAlignment(size_t byteSize, Storage storage) override;
    Ref<Heap>   newHeap(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;
    Ref<Buffer> newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset) override;

    // Prefer separate pooling/upsampling so convolution can use shared-memory
    // kernels. Direct fused pooling remains available to low-level kernel tests.
    bool isConvSupported(Fusion fusion) const override;
    bool isConcatConv2Supported(Fusion fusion) const override;
    Ref<ConcatConv2> newConcatConv2(const ConcatConvDesc& desc) override;

    Ref<Conv> newConv(const ConvDesc& desc) override;
    Ref<Pool> newPool(const PoolDesc& desc) override;
    Ref<Upsample> newUpsample(const UpsampleDesc& desc) override;
    Ref<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    Ref<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    Ref<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    Ref<ImageCopy> newImageCopy() override;
    void submitHostFunc(std::function<void()>&& f,
                        const Ref<CancellationToken>& ct = nullptr) override;
    void wait() override;

    // Convenience helpers for creating lightweight tensor views used by the
    // custom WebGPU kernels. These are not overrides of Engine::newTensor().
    WebGPUTensor makeTensor(const BufferRef& buffer, WebGPUTensorType type,
                            uint32_t n, uint32_t c, uint32_t h, uint32_t w);

    void conv2d(const WebGPUTensor& src,
                const WebGPUTensor& weight,
                const WebGPUTensor& bias,
                const WebGPUTensor& dst,
                const WebGPUConvParams& params);

    // Small placeholder buffer for unused bindings (e.g. the bias binding when
    // the bias aliases dst)
    WGPUBuffer getDummyBuffer();

    // TransferFunction carries a pointer for device-side exposure. WebGPU has
    // no device pointers, so resolve the autoexposure's stable token to a buffer.
    void registerAutoexposure(const float* token, WGPUBuffer buffer);
    void unregisterAutoexposure(const float* token);
    WGPUBuffer getAutoexposureBuffer(const float* token) const;

    // Bump-allocates a slot in a shared uniform buffer and writes the given
    // params into it (queue-ordered, so recycling slots on wrap-around cannot
    // race in-flight dispatches). Replaces a per-dispatch create/release.
    struct UniformSlot
    {
      WGPUBuffer buffer;
      size_t offset;
    };
    UniformSlot allocUniform(const void* data, size_t byteSize);

    void upsample2x(const WebGPUTensor& src,
                    const WebGPUTensor& dst);

    void pool2x2(const WebGPUTensor& src,
                 const WebGPUTensor& dst);

    void add(const WebGPUTensor& A,
             const WebGPUTensor& B,
             const WebGPUTensor& dst);

    void mul(const WebGPUTensor& A,
             const WebGPUTensor& B,
             const WebGPUTensor& dst);

    void softplus(const WebGPUTensor& src,
                  const WebGPUTensor& dst);

    void sync();

    // Command batching: ops record their compute passes into a shared pending
    // encoder; flush() finishes and submits it. Buffer reads/writes and sync()
    // flush automatically to keep queue ordering correct.
    WGPUCommandEncoder getEncoder();
    void flush() override;
    void beginExternalEncoding(WGPUCommandEncoder encoder);
    void endExternalEncoding();
    void finishExternalTextures();
    void accessTexture(WebGPUTextureBuffer* texture, bool write);
    bool isExternalEncoding() const { return externalEncoder != nullptr; }

#if defined(OIDN_WEBGPU_PROFILE)
    // Test-only GPU timestamps; caller owns the query set and resolves it after encode.
    WGPUQuerySet profileQueries = nullptr;
    std::vector<std::string> profileLabels;
    int profileConvMode = 0; // 0=optimized, 1=old kernels+fused pool, 2=old kernels+separate pool
#endif

    // Cached compute pipeline for ops with (possibly specialized) WGSL
    // sources. bindingTypes describes the buffer bindings in order.
    struct CachedPipeline
    {
      WGPUShaderModule module = nullptr;
      WGPUBindGroupLayout bindGroupLayout = nullptr;
      WGPUPipelineLayout pipelineLayout = nullptr;
      WGPUComputePipeline pipeline = nullptr;
    };
    const CachedPipeline& getPipeline(const std::string& key, const std::string& code,
                                      std::initializer_list<WGPUBufferBindingType> bindingTypes);
    const CachedPipeline& getPipeline(const std::string& key, const std::string& code,
                                      const std::vector<WGPUBindGroupLayoutEntry>& entries);

    // Instantiates a WGSL kernel template for f32 or f16 tensors: replaces the
    // $T$ placeholder with the element type, and prepends 'enable f16;' when
    // needed. Kernels accumulate in f32 in both variants.
    static std::string specializeWGSL(const char* tmpl, bool f16);

  private:
    void initAddPipeline();
    void initMulPipeline();
    void initSoftplusPipeline();

    WebGPUDevice* device;

    // Convolution variants: 0=vec4, 1=spatial tile, 2=implicit GEMM.
    // The channel constant is IC4 for variants 0/1 and IC for variant 2.
    // Pipelines are specialized per (variant, IC4, KH, KW, dtype) via override
    // constants so the loops have compile-time trip counts (crucial for
    // Tint/FXC in browsers).
    WGPUShaderModule convGemmModules[2] = {nullptr, nullptr};
    WGPUShaderModule convVec4Modules[2]  = {nullptr, nullptr}; // [f32, f16]
    WGPUShaderModule convTiledModules[2] = {nullptr, nullptr}; // [f32, f16]
    std::map<std::array<uint32_t, 5>, WGPUComputePipeline> convVariantPipelines;
    WGPUComputePipeline getConvVariantPipeline(uint32_t variant, uint32_t ic4, uint32_t kh, uint32_t kw, bool f16);

    WGPUShaderModule addShaderModule = nullptr;
    WGPUBindGroupLayout addBindGroupLayout = nullptr;
    WGPUPipelineLayout addPipelineLayout = nullptr;
    WGPUComputePipeline addPipeline = nullptr;

    WGPUShaderModule mulShaderModule = nullptr;
    WGPUBindGroupLayout mulBindGroupLayout = nullptr;
    WGPUPipelineLayout mulPipelineLayout = nullptr;
    WGPUComputePipeline mulPipeline = nullptr;

    WGPUShaderModule softplusShaderModule = nullptr;
    WGPUBindGroupLayout softplusBindGroupLayout = nullptr;
    WGPUPipelineLayout softplusPipelineLayout = nullptr;
    WGPUComputePipeline softplusPipeline = nullptr;

    WGPUBuffer dummyBuf = nullptr;
    WGPUCommandEncoder pendingEncoder = nullptr;
    WGPUCommandEncoder externalEncoder = nullptr;
    std::vector<WGPUBuffer> externalUniformPages;
    size_t externalUniformOffset = 0;
    std::unordered_set<WebGPUTextureBuffer*> textureReads, textureWrites;

    static constexpr size_t uniformArenaByteSize = 256 * 1024;
    WGPUBuffer uniformArena = nullptr;
    size_t uniformArenaOffset = 0;
    std::unordered_map<std::string, CachedPipeline> pipelineCache;
    std::unordered_map<const float*, WGPUBuffer> autoexposureBuffers;
  };

OIDN_NAMESPACE_END
