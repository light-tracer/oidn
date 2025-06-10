#pragma once
#include "core/autoexposure.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUAutoexposure final : public Autoexposure
  {
  public:
    WebGPUAutoexposure(WebGPUEngine* engine, const ImageDesc& srcDesc);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    WebGPUEngine* engine;
  };

OIDN_NAMESPACE_END
