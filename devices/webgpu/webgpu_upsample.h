#pragma once
#include "core/upsample.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUUpsample final : public Upsample
  {
  public:
    WebGPUUpsample(WebGPUEngine* engine, const UpsampleDesc& desc)
      : Upsample(desc), engine(engine) {}

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    WebGPUEngine* engine;
  };

OIDN_NAMESPACE_END
