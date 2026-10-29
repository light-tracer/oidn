#pragma once
#include "core/image.h"
#include "webgpu_engine.h"
#include <webgpu/webgpu.h>

OIDN_NAMESPACE_BEGIN

  struct WebGPUImageParams
  {
    uint32_t offset, hStride, wStride, channels;
  };

  // A WGPUBuffer view of an image's pixel data.
  // Images backed by a WebGPUBuffer are bound directly; images wrapping host
  // memory are staged through a temporary GPU buffer. For write access to a
  // host image, finish() must be called after the kernel has been submitted to
  // copy the result back to host memory.
  class WebGPUImageAccess
  {
  public:
    WebGPUImageAccess(WebGPUEngine* engine, Image* image, bool forWrite);

    WGPUBuffer getWGPUBuffer() const { return buf; }
    size_t getOffset() const { return offset; }
    size_t getByteSize() const { return bindingByteSize; }
    WebGPUImageParams getParams() const { return params; }

    void finish();

  private:
    Image* image;
    size_t byteSize;
    WGPUBuffer buf = nullptr;
    size_t offset = 0;
    size_t bindingByteSize = 0;
    WebGPUImageParams params;
    Ref<Buffer> staging; // set when the image wraps host memory
    bool forWrite;
  };

OIDN_NAMESPACE_END
