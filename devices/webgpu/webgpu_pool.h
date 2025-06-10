#pragma once
#include "core/pool.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUPool final : public Pool
  {
  public:
    WebGPUPool(WebGPUEngine* engine, const PoolDesc& desc)
      : Pool(desc), engine(engine) {}

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    WebGPUEngine* engine;
  };

OIDN_NAMESPACE_END
