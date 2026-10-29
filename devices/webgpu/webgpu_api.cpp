#include "OpenImageDenoise/oidn_webgpu.h"
#include "webgpu_device.h"
#include "webgpu_buffer.h"
#include "webgpu_texture.h"
#include "core/filter.h"

OIDN_NAMESPACE_USING
OIDN_API_NAMESPACE_BEGIN

namespace
{
  template<typename T, typename F>
  T guarded(Device* device, T failure, F&& function)
  {
    try { return function(); }
    catch (const Exception& e) { Device::setError(device, e.code(), e.what()); }
    catch (const std::bad_alloc&) { Device::setError(device, Error::OutOfMemory, "out of memory"); }
    catch (const std::exception& e) { Device::setError(device, Error::Unknown, e.what()); }
    catch (...) { Device::setError(device, Error::Unknown, "unknown WebGPU error"); }
    return failure;
  }

  WebGPUDevice* webgpuDevice(Device* device)
  {
    if (!device || device->getType() != DeviceType::WGPU)
      throw Exception(Error::InvalidArgument, "a WebGPU device is required");
    device->checkCommitted();
    return static_cast<WebGPUDevice*>(device);
  }
}

OIDN_WEBGPU_API OIDNDevice oidnNewWebGPUDevice(const OIDNWebGPUContextDesc* desc)
{
  return guarded<OIDNDevice>(nullptr, nullptr, [&]()
  {
    if (!desc || desc->structSize < sizeof(*desc))
      throw Exception(Error::InvalidArgument, "invalid WebGPU context descriptor");
    Ref<Device> device = makeRef<WebGPUDevice>(desc->device, desc->queue);
    return reinterpret_cast<OIDNDevice>(device.detach());
  });
}

OIDN_WEBGPU_API OIDNBuffer oidnNewSharedBufferFromWebGPU(OIDNDevice handle, const OIDNWebGPUBufferDesc* desc)
{
  Device* device = reinterpret_cast<Device*>(handle);
  return guarded<OIDNBuffer>(device, nullptr, [&]()
  {
    auto* gpu = webgpuDevice(device);
    std::lock_guard<std::mutex> lock(device->getMutex());
    if (!desc || desc->structSize < sizeof(*desc) || !desc->buffer)
      throw Exception(Error::InvalidArgument, "invalid WebGPU buffer descriptor");
    const uint64_t total = wgpuBufferGetSize(desc->buffer);
    if (desc->byteOffset > total || total > SIZE_MAX)
      throw Exception(Error::InvalidArgument, "WebGPU buffer offset/size is out of range");
    const size_t size = desc->byteSize ? desc->byteSize : size_t(total) - desc->byteOffset;
    auto* engine = static_cast<WebGPUEngine*>(gpu->getEngine());
    Ref<Buffer> buffer = makeRef<WebGPUBuffer>(engine, desc->buffer, desc->byteOffset, size);
    buffer->toUser();
    return reinterpret_cast<OIDNBuffer>(buffer.detach());
  });
}

OIDN_WEBGPU_API bool oidnPrepareFilterWebGPU(OIDNFilter handle)
{
  auto* filter = reinterpret_cast<Filter*>(handle);
  Device* device = filter ? filter->getDevice() : nullptr;
  return guarded<bool>(device, false, [&]()
  {
    auto* gpu = webgpuDevice(device);
    std::lock_guard<std::mutex> lock(device->getMutex());
    if (static_cast<WebGPUEngine*>(gpu->getEngine())->isExternalEncoding())
      throw Exception(Error::InvalidOperation, "cannot prepare during an encode");
    gpu->beginPrepare();
    try { filter->commit(); }
    catch (...) { gpu->endPrepare(); throw; }
    gpu->endPrepare();
    return true;
  });
}

OIDN_WEBGPU_API bool oidnSetFilterImageFromWebGPUTexture(
  OIDNFilter handle, const char* name, const OIDNWebGPUTextureDesc* desc)
{
  auto* filter = reinterpret_cast<Filter*>(handle);
  Device* device = filter ? filter->getDevice() : nullptr;
  return guarded<bool>(device, false, [&]()
  {
    auto* gpu = webgpuDevice(device);
    std::lock_guard<std::mutex> lock(device->getMutex());
    if (!name || !desc || desc->structSize < sizeof(*desc))
      throw Exception(Error::InvalidArgument, "invalid WebGPU texture descriptor");
    if (std::strcmp(name, "color") && std::strcmp(name, "albedo") &&
        std::strcmp(name, "normal") && std::strcmp(name, "output"))
      throw Exception(Error::InvalidArgument, "unknown WebGPU image name");
    auto* engine = static_cast<WebGPUEngine*>(gpu->getEngine());
    Ref<Buffer> buffer = makeRef<WebGPUTextureBuffer>(engine, *desc, std::strcmp(name, "output") == 0);
    auto image = makeRef<Image>(buffer, Format::Float3, desc->width, desc->height, 0, 16, 0);
    filter->setImage(name, image);
    return true;
  });
}

OIDN_WEBGPU_API bool oidnEncodeFilterWebGPU(OIDNFilter handle, WGPUCommandEncoder encoder)
{
  auto* filter = reinterpret_cast<Filter*>(handle);
  Device* device = filter ? filter->getDevice() : nullptr;
  return guarded<bool>(device, false, [&]()
  {
    auto* gpu = webgpuDevice(device);
    std::lock_guard<std::mutex> lock(device->getMutex());
    auto* engine = static_cast<WebGPUEngine*>(gpu->getEngine());
    engine->beginExternalEncoding(encoder);
    try { filter->execute(SyncMode::Async); engine->finishExternalTextures(); }
    catch (...) { engine->endExternalEncoding(); throw; }
    engine->endExternalEncoding();
    return true;
  });
}

OIDN_API_NAMESPACE_END
