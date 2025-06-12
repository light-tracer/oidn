#pragma once
#include "core/engine.h"
#include "OpenImageDenoise/webgpu.h"
#include "OpenImageDenoise/oidn.hpp"
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

    void conv2d_eltwise(const WebGPUTensor& src,
                        const WebGPUTensor& weight,
                        const WebGPUTensor& bias,
                        const WebGPUTensor& dst);

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
