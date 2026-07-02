#include "webgpu_image_util.h"
#include "webgpu_buffer.h"

OIDN_NAMESPACE_BEGIN

  WebGPUImageAccess::WebGPUImageAccess(WebGPUEngine* engine, Image* image,
                                       size_t byteSize, bool forWrite)
    : engine(engine), image(image), byteSize(byteSize), forWrite(forWrite)
  {
    auto* wb = image ? dynamic_cast<WebGPUBuffer*>(image->getBuffer()) : nullptr;
    if (wb)
    {
      buf    = wb->getWGPUBuffer();
      offset = wb->getWGPUBufferOffset() + image->getByteOffset();
      if (offset % 256 != 0)
        throw Exception(Error::InvalidArgument,
          "WebGPU image offset is not 256-byte aligned");
    }
    else if (image && image->getPtr())
    {
      // Host memory: stage through a GPU buffer
      staging = engine->newBuffer(byteSize, Storage::Device);
      if (!forWrite)
        staging->write(0, byteSize, image->getPtr());
      buf    = static_cast<WebGPUBuffer*>(staging.get())->getWGPUBuffer();
      offset = 0;
    }
    else
      throw Exception(Error::InvalidArgument, "image has no accessible storage");
  }

  void WebGPUImageAccess::finish()
  {
    if (forWrite && staging)
      staging->read(0, byteSize, image->getPtr());
  }

OIDN_NAMESPACE_END
