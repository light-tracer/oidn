#include "webgpu_arena.h"
#include "webgpu_engine.h"

OIDN_NAMESPACE_BEGIN

  WebGPUArena::WebGPUArena(WebGPUEngine* engine, size_t byteSize)
    : engine(engine), byteSize(byteSize)
  {
    heap = makeRef<WebGPUHeap>(engine, byteSize, Storage::Device);
  }

  Ref<Buffer> WebGPUArena::newBuffer(size_t byteSize, size_t byteOffset)
  {
    return engine->newBuffer(this, byteSize, byteOffset);
  }

OIDN_NAMESPACE_END
