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

    // Buffer
    Ref<Buffer> newBuffer(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;

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

    WebGPUTensor newTensor(const float* data, WebGPUTensorType type,
                           uint32_t n, uint32_t c, uint32_t h, uint32_t w);
    WebGPUTensor newTensor(const BufferRef& buffer, WebGPUTensorType type,
                           uint32_t n, uint32_t c, uint32_t h, uint32_t w);

    void conv2d_eltwise(const WebGPUTensor& src,
                        const WebGPUTensor& weight,
                        const WebGPUTensor& bias,
                        const WebGPUTensor& dst);

    void upsample2x(const WebGPUTensor& src,
                    const WebGPUTensor& dst);

    void pool2x2(const WebGPUTensor& src,
                 const WebGPUTensor& dst);

    void sync();

  private:
    void initPipeline();
    void initUpsamplePipeline();
    void initPoolPipeline();

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
