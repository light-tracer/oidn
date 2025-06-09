#pragma once
#include "core/engine.h"
#include "OpenImageDenoise/webgpu.h"
#include <vector>
#include <unordered_map>

OIDN_NAMESPACE_BEGIN

  class WebGPUDevice;

  class WebGPUEngine : public Engine
  {
  public:
    explicit WebGPUEngine(WebGPUDevice* device);
    ~WebGPUEngine();

    Device* getDevice() const override;

    WebGPUTensor newTensor(const float* data, WebGPUTensorType type,
                           uint32_t n, uint32_t c, uint32_t h, uint32_t w);

    void conv2d_eltwise(const WebGPUTensor& src,
                        const WebGPUTensor& weight,
                        const WebGPUTensor& bias,
                        const WebGPUTensor& dst);

    void sync();

  private:
    void initPipeline();

    WebGPUDevice* device;

    WGPUShaderModule shaderModule = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUComputePipeline pipeline = nullptr;

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
