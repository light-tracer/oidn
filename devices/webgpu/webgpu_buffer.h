#pragma once
#include "core/buffer.h"
#include "webgpu_engine.h"
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  class WebGPUDevice;
  class WebGPUEngine;

  class WebGPUBuffer : public Buffer
  {
  public:
    WebGPUBuffer(WebGPUEngine* engine, size_t byteSize);
    ~WebGPUBuffer();

    Engine* getEngine() const override { return engine; }
    WGPUBuffer getWGPUBuffer() const { return buffer; }
    void* getPtr() const override { return nullptr; }
    void* getHostPtr() const override { return nullptr; }
    size_t getByteSize() const override { return byteSize; }
    bool isShared() const override { return false; }
    Storage getStorage() const override { return Storage::Device; }

    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr,
              SyncMode sync = SyncMode::Blocking) override;
    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr,
               SyncMode sync = SyncMode::Blocking) override;

  private:
    void init();
    void free();

    WebGPUEngine* engine;
    WGPUBuffer buffer;
    size_t byteSize;
  };

OIDN_NAMESPACE_END
