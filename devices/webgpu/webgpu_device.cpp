#include "webgpu_device.h"
#include "webgpu_engine.h"
#include "core/context.h"
#include <cstring>
#include <iostream>

#if defined(__EMSCRIPTEN__)
  #include <emscripten.h>
#else
  // wgpu-native extension header (wgpuDevicePoll, wgpuSetLogCallback, ...);
  // not available in the browser (emdawnwebgpu) build
  #include <webgpu/wgpu.h>
#endif

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

#if !defined(__EMSCRIPTEN__)
  static void handle_log(WGPULogLevel level, WGPUStringView message, void*)
  {
    fprintf(stderr, "wgpu log (%d): %.*s\n", (int)level,
            (int)message.length, message.data);
    fflush(stderr);
  }
#endif

  // Advance async work. On native this just drains the instance event queue; on
  // the web it also yields to the browser event loop (via Asyncify) so pending
  // promises can resolve.
  static void pumpEvents(WGPUInstance instance)
  {
  #if defined(__EMSCRIPTEN__)
    emscripten_sleep(1);
  #endif
    wgpuInstanceProcessEvents(instance);
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
    opts.powerPreference = WGPUPowerPreference_HighPerformance;
    WGPURequestAdapterCallbackInfo adapterCb{};
    adapterCb.mode = WGPUCallbackMode_AllowProcessEvents;
    adapterCb.callback = handle_request_adapter;
    adapterCb.userdata1 = &adapter;
    wgpuInstanceRequestAdapter(instance, &opts, adapterCb);
    while (!adapter)
      pumpEvents(instance);

    if (isVerbose())
    {
      WGPUAdapterInfo info{};
      if (wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success)
      {
        std::cout << "  Adapter : ";
        std::cout.write(info.device.data, info.device.length);
        std::cout << std::endl;
        wgpuAdapterInfoFreeMembers(info);
      }
    }

  #if !defined(__EMSCRIPTEN__)
    if (isVerbose(2))
    {
      wgpuSetLogCallback(handle_log, nullptr);
      wgpuSetLogLevel(WGPULogLevel_Warn);
    }
  #endif

    WGPUDeviceDescriptor devDesc{};
    devDesc.uncapturedErrorCallbackInfo.callback = handle_uncaptured_error;

    WGPURequestDeviceCallbackInfo deviceCb{};
    deviceCb.mode = WGPUCallbackMode_AllowProcessEvents;
    deviceCb.callback = handle_request_device;
    deviceCb.userdata1 = &device;
    wgpuAdapterRequestDevice(adapter, &devDesc, deviceCb);
    while (!device)
      pumpEvents(instance);

    queue = wgpuDeviceGetQueue(device);

    // HWC tensors (like Metal); OHWI weights so that input channels are
    // contiguous in memory, enabling vec4 loads in the conv kernel
    tensorDataType = DataType::Float32;
    weightDataType = DataType::Float32;
    tensorLayout   = TensorLayout::hwc;
    weightLayout   = TensorLayout::ohwi;
    tensorBlockC   = 1;

    subdevices.emplace_back(new Subdevice(std::unique_ptr<Engine>(new WebGPUEngine(this))));
  }

  void WebGPUDevice::submit(WGPUCommandBuffer cmdBuf)
  {
    wgpuQueueSubmit(queue, 1, &cmdBuf);
  }

  void WebGPUDevice::sync()
  {
  #if defined(__EMSCRIPTEN__)
    // There is no blocking device poll in the browser; yield to the event loop
    // (via Asyncify) so the queue drains and pending map callbacks fire
    emscripten_sleep(1);
    wgpuInstanceProcessEvents(instance);
  #else
    wgpuDevicePoll(device, true, nullptr);
  #endif
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
