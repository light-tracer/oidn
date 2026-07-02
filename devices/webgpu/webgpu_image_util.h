#pragma once
#include "core/image.h"
#include "webgpu_engine.h"
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  // A WGPUBuffer view of an image's pixel data.
  // Images backed by a WebGPUBuffer are bound directly; images wrapping host
  // memory are staged through a temporary GPU buffer. For write access to a
  // host image, finish() must be called after the kernel has been submitted to
  // copy the result back to host memory.
  class WebGPUImageAccess
  {
  public:
    WebGPUImageAccess(WebGPUEngine* engine, Image* image, size_t byteSize, bool forWrite);

    WGPUBuffer getWGPUBuffer() const { return buf; }
    size_t getOffset() const { return offset; }

    void finish();

  private:
    WebGPUEngine* engine;
    Image* image;
    size_t byteSize;
    WGPUBuffer buf = nullptr;
    size_t offset = 0;
    Ref<Buffer> staging; // set when the image wraps host memory
    bool forWrite;
  };

OIDN_NAMESPACE_END
