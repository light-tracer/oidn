#include "webgpu_heap.h"
#include "webgpu_device.h"

OIDN_NAMESPACE_BEGIN

  WebGPUHeap::WebGPUHeap(WebGPUEngine* engine, size_t byteSize, Storage storage)
    : engine(engine), buffer(nullptr), byteSize(byteSize), storage(storage)
  {
    if (storage == Storage::Undefined)
      this->storage = Storage::Device;
    init();
  }

  WebGPUHeap::~WebGPUHeap()
  {
    free();
  }

  void WebGPUHeap::init()
  {
    if (byteSize == 0)
      return;

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    buffer = dev->createBuffer(byteSize,
                               WGPUBufferUsage_Storage |
                               WGPUBufferUsage_CopySrc |
                               WGPUBufferUsage_CopyDst);
    if (!buffer)
      throw Exception(Error::OutOfMemory, "failed to create WebGPU heap buffer");
  }

  void WebGPUHeap::free()
  {
    if (buffer)
      wgpuBufferRelease(buffer);
    buffer = nullptr;
  }

  void WebGPUHeap::realloc(size_t newByteSize)
  {
    if (newByteSize == byteSize)
      return;

    preRealloc();
    free();
    byteSize = newByteSize;
    init();
    postRealloc();
  }

OIDN_NAMESPACE_END
