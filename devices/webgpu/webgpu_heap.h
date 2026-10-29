#pragma once
#include "core/heap.h"
#include "webgpu_engine.h"
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  class WebGPUHeap : public Heap
  {
    friend class WebGPUBuffer;

  public:
    WebGPUHeap(WebGPUEngine* engine, size_t byteSize, Storage storage);
    ~WebGPUHeap();

    Engine* getEngine() const override { return engine; }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return storage; }

    WGPUBuffer getWGPUBuffer() const { return buffer; }

    void realloc(size_t newByteSize) override;

  private:
    void init();
    void free();

    WebGPUEngine* engine;
    WGPUBuffer buffer;
    size_t byteSize;
    Storage storage;
  };

OIDN_NAMESPACE_END
