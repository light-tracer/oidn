#include "core/context.h"
#include "webgpu_device.h"

OIDN_NAMESPACE_BEGIN

  class WebGPUDeviceFactory : public DeviceFactory
  {
  public:
    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      return makeRef<WebGPUDevice>();
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_webgpu)
  {
    Context::registerDeviceType<WebGPUDeviceFactory>(DeviceType::WGPU,
                                                     {makeRef<PhysicalDevice>(DeviceType::WGPU,0)});
  }

OIDN_NAMESPACE_END
