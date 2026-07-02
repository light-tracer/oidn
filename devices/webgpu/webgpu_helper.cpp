#include "webgpu_engine.h"
#include "webgpu_device.h"

OIDN_NAMESPACE_BEGIN

  OIDN_WEBGPU_API WebGPUEngine* getEngine(const DeviceRef& device)
  {
    if (!device)
      return nullptr;
    Device* impl = reinterpret_cast<Device*>(device.getHandle());
    return static_cast<WebGPUEngine*>(impl->getEngine());
  }

OIDN_NAMESPACE_END
