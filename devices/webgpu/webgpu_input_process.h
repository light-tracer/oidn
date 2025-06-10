#pragma once
#include "core/input_process.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUInputProcess final : public InputProcess
  {
  public:
    WebGPUInputProcess(WebGPUEngine* engine, const InputProcessDesc& desc);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    WebGPUEngine* engine;
  };

OIDN_NAMESPACE_END
