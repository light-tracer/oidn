#pragma once
#include "core/arena.h"
#include "webgpu_heap.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUArena final : public Arena
  {
  public:
    WebGPUArena(WebGPUEngine* engine, size_t byteSize);

    Engine* getEngine() const override { return engine; }
    Heap* getHeap() const override { return heap.get(); }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return heap->getStorage(); }

    Ref<Buffer> newBuffer(size_t byteSize, size_t byteOffset) override;

  private:
    WebGPUEngine* engine;
    Ref<WebGPUHeap> heap;
    size_t byteSize;
  };

OIDN_NAMESPACE_END
