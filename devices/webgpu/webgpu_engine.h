#pragma once
#include "core/engine.h"
#include "OpenImageDenoise/webgpu.h"
#include "OpenImageDenoise/oidn.hpp"
#include <array>
#include <map>
#include <vector>
#include <unordered_map>

OIDN_NAMESPACE_BEGIN

  class WebGPUDevice;
  class WebGPUInputProcess;
  class WebGPUOutputProcess;
  class WebGPUImageCopy;
  class WebGPUAutoexposure;

  class WebGPUEngine : public Engine
  {
  public:
    explicit WebGPUEngine(WebGPUDevice* device);
    ~WebGPUEngine();

    Device* getDevice() const override;

    // Heap / Buffer
    SizeAndAlignment getBufferByteSizeAndAlignment(size_t byteSize, Storage storage) override;
    Ref<Heap>   newHeap(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;
    Ref<Buffer> newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset) override;

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
    WebGPUTensor makeTensor(const float* data, WebGPUTensorType type,
                            uint32_t n, uint32_t c, uint32_t h, uint32_t w);
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
    void flush();

    // Cached compute pipeline for ops with static WGSL sources (keyed by the
    // source pointer). bindingTypes describes the buffer bindings in order.
    struct CachedPipeline
    {
      WGPUShaderModule module = nullptr;
      WGPUBindGroupLayout bindGroupLayout = nullptr;
      WGPUPipelineLayout pipelineLayout = nullptr;
      WGPUComputePipeline pipeline = nullptr;
    };
    const CachedPipeline& getPipeline(const char* code,
                                      std::initializer_list<WGPUBufferBindingType> bindingTypes);

  private:
    void initPipeline();
    void initUpsamplePipeline();
    void initPoolPipeline();
    void initAddPipeline();
    void initMulPipeline();
    void initSoftplusPipeline();

    WebGPUDevice* device;

    WGPUShaderModule shaderModule = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUComputePipeline pipeline = nullptr;

    // Vectorized conv variant (input channels multiple of 4); shares
    // bindGroupLayout/pipelineLayout with the scalar conv pipeline. Pipelines
    // are specialized per (IC4, KH, KW) via override constants so the loops
    // have compile-time trip counts (crucial for Tint/FXC in browsers).
    WGPUShaderModule convVec4ShaderModule = nullptr;
    std::map<std::array<uint32_t, 3>, WGPUComputePipeline> convVec4Pipelines;
    WGPUComputePipeline getConvVec4Pipeline(uint32_t ic4, uint32_t kh, uint32_t kw);

    WGPUShaderModule upsampleShaderModule = nullptr;
    WGPUBindGroupLayout upsampleBindGroupLayout = nullptr;
    WGPUPipelineLayout upsamplePipelineLayout = nullptr;
    WGPUComputePipeline upsamplePipeline = nullptr;

    WGPUShaderModule poolShaderModule = nullptr;
    WGPUBindGroupLayout poolBindGroupLayout = nullptr;
    WGPUPipelineLayout poolPipelineLayout = nullptr;
    WGPUComputePipeline poolPipeline = nullptr;

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
    std::unordered_map<const char*, CachedPipeline> pipelineCache;

    struct PendingReadback
    {
      WGPUBuffer src;
      WGPUBuffer readback;
      void* host;
      size_t size;
    };
    std::vector<PendingReadback> readbacks;

    std::unordered_map<WGPUBuffer, void*> outputHosts;
  };

OIDN_NAMESPACE_END
