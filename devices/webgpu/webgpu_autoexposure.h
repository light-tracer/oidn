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

    // WebGPU buffers are not host-accessible, so the result is kept in a
    // host-side copy that the transfer function can dereference
    float* getDstPtr() const override { return &hostResult; }

  private:
    WebGPUEngine* engine;
    mutable float hostResult = 1.f;
  };

OIDN_NAMESPACE_END
