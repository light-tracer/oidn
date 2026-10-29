#pragma once
#include "core/autoexposure.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUAutoexposure final : public Autoexposure
  {
  public:
    WebGPUAutoexposure(WebGPUEngine* engine, const ImageDesc& srcDesc);
    ~WebGPUAutoexposure();

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

    // Stable token resolved by WebGPU input/output processing to scaleBuffer.
    // The scalar is only a fallback; execution never reads the exposure on CPU.
    float* getDstPtr() const override { return &hostResult; }
    Ref<Buffer> getResultBuffer() const { return scaleBuffer; }

  private:
    WebGPUEngine* engine;
    Ref<Buffer> binsBuffer;
    Ref<Buffer> scaleBuffer;
    mutable float hostResult = 1.f;
  };

OIDN_NAMESPACE_END
