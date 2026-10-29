#include "webgpu_image_util.h"
#include "webgpu_buffer.h"
#include "webgpu_device.h"
#include "webgpu_texture.h"

OIDN_NAMESPACE_BEGIN

  WebGPUImageAccess::WebGPUImageAccess(WebGPUEngine* engine, Image* image, bool forWrite)
    : image(image), byteSize(image ? image->getByteSize() : 0), forWrite(forWrite)
  {
    if (!image || image->getDataType() != DataType::Float32 ||
        image->getC() < 1 || image->getC() > 3)
      throw Exception(Error::InvalidArgument, "WebGPU images require 1-3 float32 channels");

    const auto& desc = image->getDesc();
    if (desc.hByteStride % 4 || desc.wByteStride % 4)
      throw Exception(Error::InvalidArgument, "WebGPU image strides must be float-aligned");

    size_t dataOffset = 0;
    auto* wb = image ? dynamic_cast<WebGPUBuffer*>(image->getBuffer()) : nullptr;
    if (wb)
    {
      if (auto* texture = dynamic_cast<WebGPUTextureBuffer*>(wb))
        engine->accessTexture(texture, forWrite);
      buf    = wb->getWGPUBuffer();
      dataOffset = wb->getWGPUBufferOffset() + image->getByteOffset();
      if (dataOffset % 4 != 0)
        throw Exception(Error::InvalidArgument,
          "WebGPU image offset is not float-aligned");
      offset = dataOffset & ~size_t(255);
      dataOffset -= offset;
    }
    else if (image && image->getPtr())
    {
      if (static_cast<WebGPUDevice*>(engine->getDevice())->isExternal())
        throw Exception(Error::InvalidArgument, "external WebGPU encoding requires GPU-backed images");
      // Host memory: stage through a GPU buffer
      staging = engine->newBuffer(byteSize, Storage::Device);
      // Output kernels may only touch one tile. Preserve the other pixels and
      // padding when staging an image which is subsequently copied back whole.
      staging->write(0, byteSize, image->getPtr());
      buf    = static_cast<WebGPUBuffer*>(staging.get())->getWGPUBuffer();
      offset = 0;
    }
    else
      throw Exception(Error::InvalidArgument, "image has no accessible storage");

    bindingByteSize = roundByteSize4(dataOffset + byteSize);
    if (bindingByteSize / 4 > UINT32_MAX || desc.hByteStride / 4 > UINT32_MAX ||
        desc.wByteStride / 4 > UINT32_MAX)
      throw Exception(Error::InvalidArgument, "WebGPU image address exceeds shader limits");
    params = {uint32_t(dataOffset / 4), uint32_t(desc.hByteStride / 4),
              uint32_t(desc.wByteStride / 4), uint32_t(image->getC())};
  }

  void WebGPUImageAccess::finish()
  {
    if (forWrite && staging)
      staging->read(0, byteSize, image->getPtr());
  }

OIDN_NAMESPACE_END
