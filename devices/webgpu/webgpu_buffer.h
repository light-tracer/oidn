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
    WebGPUBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset);
    ~WebGPUBuffer();

    Engine* getEngine() const override { return engine; }
    WGPUBuffer getWGPUBuffer() const { return buffer; }

    // Offset of this buffer within the backing WGPUBuffer (nonzero for arena
    // suballocations, where the WGPUBuffer belongs to the heap). Must be added
    // to any per-tensor/image offset when binding.
    size_t getWGPUBufferOffset() const { return byteOffset; }
    void* getPtr() const override { return nullptr; }
    void* getHostPtr() const override { return nullptr; }
    size_t getByteSize() const override { return byteSize; }
    bool isShared() const override { return shared; }
    Storage getStorage() const override { return Storage::Device; }

    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr,
              SyncMode sync = SyncMode::Blocking) override;
    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr,
               SyncMode sync = SyncMode::Blocking) override;

  protected:
    void postRealloc() override;

  private:
    void init();
    void free();

    WebGPUEngine* engine;
    WGPUBuffer buffer;
    size_t byteSize;
    bool shared;
  };

OIDN_NAMESPACE_END
