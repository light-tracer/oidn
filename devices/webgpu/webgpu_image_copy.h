#pragma once
#include "core/image_copy.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUImageCopy final : public ImageCopy
  {
  public:
    explicit WebGPUImageCopy(WebGPUEngine* engine);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    WebGPUEngine* engine;
  };

OIDN_NAMESPACE_END
