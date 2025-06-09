#include "webgpu_engine.h"
#include "webgpu_device.h"

OIDN_NAMESPACE_BEGIN

  WebGPUEngine* getEngine(DeviceRef device)
  {
    if (!device)
      return nullptr;
    return static_cast<WebGPUEngine*>(device->getEngine());
  }

OIDN_NAMESPACE_END
