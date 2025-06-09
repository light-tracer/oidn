#pragma once
#include "core/device.h"
#include "webgpu_engine.h"
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  class WebGPUDevice : public Device
  {
  public:
    WebGPUDevice();
    ~WebGPUDevice();

    DeviceType getType() const override { return DeviceType::WGPU; }

    void submit(WGPUCommandBuffer cmdBuf);
    void sync();

    WGPUBuffer createBuffer(size_t byteSize, WGPUBufferUsageFlags usage,
                            const void* initData = nullptr);

  protected:
    void init() override;
    void wait() override { sync(); }

  public:
    WGPUInstance instance = nullptr;
    WGPUAdapter  adapter  = nullptr;
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
  };

OIDN_NAMESPACE_END
