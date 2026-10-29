#pragma once
#include "core/conv.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUConv final : public Conv
  {
  public:
    WebGPUConv(WebGPUEngine* engine, const ConvDesc& desc);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    WebGPUEngine* engine;
  };

OIDN_NAMESPACE_END
