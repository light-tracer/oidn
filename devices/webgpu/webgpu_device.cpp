#include "webgpu_device.h"
#include "webgpu_engine.h"
#include "core/context.h"
#include <cstring>

OIDN_NAMESPACE_BEGIN

  static void handle_request_adapter(WGPURequestAdapterStatus status,
                                     WGPUAdapter adapter,
                                     const char* message,
                                     void* userdata)
  {
    auto* out = static_cast<WGPUAdapter*>(userdata);
    if (status == WGPURequestAdapterStatus_Success)
      *out = adapter;
  }

  static void handle_request_device(WGPURequestDeviceStatus status,
                                    WGPUDevice device,
                                    const char* message,
                                    void* userdata)
  {
    auto* out = static_cast<WGPUDevice*>(userdata);
    if (status == WGPURequestDeviceStatus_Success)
      *out = device;
  }

  WebGPUDevice::WebGPUDevice() {}

  WebGPUDevice::~WebGPUDevice()
  {
    if (queue)   wgpuQueueRelease(queue);
    if (device)  wgpuDeviceRelease(device);
    if (adapter) wgpuAdapterRelease(adapter);
    if (instance) wgpuInstanceRelease(instance);
  }

  void WebGPUDevice::init()
  {
    instance = wgpuCreateInstance(nullptr);
    if (!instance)
      throw std::runtime_error("failed to create WebGPU instance");

    WGPURequestAdapterOptions opts{};
    bool done = false;
    wgpuInstanceRequestAdapter(instance, &opts,
        (WGPURequestAdapterCallbackInfo){handle_request_adapter, &adapter, nullptr});
    while (!adapter)
      wgpuInstanceProcessEvents(instance);

    wgpuAdapterRequestDevice(adapter, nullptr,
        (WGPURequestDeviceCallbackInfo){handle_request_device, &device, nullptr});
    while (!device)
      wgpuInstanceProcessEvents(instance);

    queue = wgpuDeviceGetQueue(device);

    subdevices.emplace_back(new Subdevice(std::unique_ptr<Engine>(new WebGPUEngine(this))));
  }

  void WebGPUDevice::submit(WGPUCommandBuffer cmdBuf)
  {
    wgpuQueueSubmit(queue, 1, &cmdBuf);
  }

  void WebGPUDevice::sync()
  {
    wgpuDevicePoll(device, true, nullptr);
  }

  WGPUBuffer WebGPUDevice::createBuffer(size_t byteSize, WGPUBufferUsage usage,
                                        const void* initData)
  {
    WGPUBufferDescriptor desc{};
    desc.usage = usage;
    desc.size  = byteSize;
    desc.mappedAtCreation = initData != nullptr;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
    if (initData)
    {
      std::memcpy(wgpuBufferGetMappedRange(buf, 0, byteSize), initData, byteSize);
      wgpuBufferUnmap(buf);
    }
    return buf;
  }

OIDN_NAMESPACE_END
