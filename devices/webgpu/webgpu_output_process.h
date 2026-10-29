#pragma once
#include "core/output_process.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUOutputProcess final : public OutputProcess
  {
  public:
    WebGPUOutputProcess(WebGPUEngine* engine, const OutputProcessDesc& desc);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    WebGPUEngine* engine;
  };

OIDN_NAMESPACE_END
