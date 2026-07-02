#include "webgpu_device.h"
#include "webgpu_engine.h"
#include "core/context.h"
#include <cstring>
#include <webgpu/wgpu.h>

OIDN_NAMESPACE_BEGIN

  static void handle_request_adapter(WGPURequestAdapterStatus status,
                                     WGPUAdapter adapter,
                                     WGPUStringView,
                                     void* userdata1,
                                     void*)
  {
    auto* out = static_cast<WGPUAdapter*>(userdata1);
    if (status == WGPURequestAdapterStatus_Success)
      *out = adapter;
  }

  static void handle_request_device(WGPURequestDeviceStatus status,
                                    WGPUDevice device,
                                    WGPUStringView,
                                    void* userdata1,
                                    void*)
  {
    auto* out = static_cast<WGPUDevice*>(userdata1);
    if (status == WGPURequestDeviceStatus_Success)
      *out = device;
  }

  static void handle_uncaptured_error(const WGPUDevice*, WGPUErrorType type,
                                      WGPUStringView message, void*, void*)
  {
    fprintf(stderr, "WebGPU error (%d): %.*s\n", (int)type,
            (int)message.length, message.data);
    fflush(stderr);
  }

  static void handle_log(WGPULogLevel level, WGPUStringView message, void*)
  {
    fprintf(stderr, "wgpu log (%d): %.*s\n", (int)level,
            (int)message.length, message.data);
    fflush(stderr);
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
    WGPURequestAdapterCallbackInfo adapterCb{};
    adapterCb.mode = WGPUCallbackMode_AllowProcessEvents;
    adapterCb.callback = handle_request_adapter;
    adapterCb.userdata1 = &adapter;
    wgpuInstanceRequestAdapter(instance, &opts, adapterCb);
    while (!adapter)
      wgpuInstanceProcessEvents(instance);

    if (isVerbose(2))
    {
      wgpuSetLogCallback(handle_log, nullptr);
      wgpuSetLogLevel(WGPULogLevel_Warn);
    }

    WGPUDeviceDescriptor devDesc{};
    devDesc.uncapturedErrorCallbackInfo.callback = handle_uncaptured_error;

    WGPURequestDeviceCallbackInfo deviceCb{};
    deviceCb.mode = WGPUCallbackMode_AllowProcessEvents;
    deviceCb.callback = handle_request_device;
    deviceCb.userdata1 = &device;
    wgpuAdapterRequestDevice(adapter, &devDesc, deviceCb);
    while (!device)
      wgpuInstanceProcessEvents(instance);

    queue = wgpuDeviceGetQueue(device);

    // Use the same native tensor layouts as the Metal backend (NHWC / OIHW)
    tensorDataType = DataType::Float32;
    weightDataType = DataType::Float32;
    tensorLayout   = TensorLayout::hwc;
    weightLayout   = TensorLayout::oihw;
    tensorBlockC   = 1;

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
