#pragma once
#include "core/device.h"
#include "webgpu_engine.h"
#include "webgpu_compat.h"
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  class WebGPUDevice : public Device
  {
  public:
    WebGPUDevice();
    WebGPUDevice(WGPUDevice device, WGPUQueue queue);
    ~WebGPUDevice();

    DeviceType getType() const override { return DeviceType::WGPU; }

    void submit(WGPUCommandBuffer cmdBuf);
    void sync();
    void waitForRelease() override { if (external) flush(); else wait(); }
    bool isExternal() const { return external; }
    void beginPrepare();
    void endPrepare() { preparing = false; }

    // Blocks until the future completes (web builds only; no-op on native,
    // where the blocking polls in sync() are used instead)
    void waitFuture(WGPUFuture future);

    WGPUBuffer createBuffer(size_t byteSize, OIDNWGPUBufferUsage usage,
                            const void* initData = nullptr);

  protected:
    void init() override;
    void initExternal();
    void wait() override;
    void flush() override;

  public:
    bool external = false;
    bool preparing = false;
    WGPUInstance instance = nullptr;
    WGPUAdapter  adapter  = nullptr;
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
  };

OIDN_NAMESPACE_END
