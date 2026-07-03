#include "webgpu_device.h"
#include "webgpu_engine.h"
#include "core/context.h"
#include <cstring>
#include <iostream>

#if !defined(__EMSCRIPTEN__)
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
  #if defined(__EMSCRIPTEN__)
    // The web build blocks on async operations with wgpuInstanceWaitAny
    // (awaiting the underlying promise via Asyncify), which must be enabled at
    // instance creation. Polling with timers instead would be subject to the
    // browser throttling timers in background tabs (up to ~1 s per tick).
    WGPUInstanceDescriptor instDesc{};
    static const WGPUInstanceFeatureName timedWaitAny = WGPUInstanceFeatureName_TimedWaitAny;
    instDesc.requiredFeatureCount = 1;
    instDesc.requiredFeatures = &timedWaitAny;
    instance = wgpuCreateInstance(&instDesc);
  #else
    instance = wgpuCreateInstance(nullptr);
  #endif
    if (!instance)
      throw std::runtime_error("failed to create WebGPU instance");

    WGPURequestAdapterOptions opts{};
    opts.powerPreference = WGPUPowerPreference_HighPerformance;
    WGPURequestAdapterCallbackInfo adapterCb{};
    adapterCb.callback = handle_request_adapter;
    adapterCb.userdata1 = &adapter;
  #if defined(__EMSCRIPTEN__)
    adapterCb.mode = WGPUCallbackMode_WaitAnyOnly;
    waitFuture(wgpuInstanceRequestAdapter(instance, &opts, adapterCb));
  #else
    adapterCb.mode = WGPUCallbackMode_AllowProcessEvents;
    wgpuInstanceRequestAdapter(instance, &opts, adapterCb);
    while (!adapter)
      wgpuInstanceProcessEvents(instance);
  #endif
    if (!adapter)
      throw std::runtime_error("failed to get WebGPU adapter");

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

    // The defaults (256 MB max buffer, 128 MB max storage binding) are too
    // small for the scratch heap and tensor bindings of larger images, so
    // request everything the adapter supports (requesting the full supported
    // limits struct is valid by construction on both wgpu-native and Dawn)
    WGPULimits requiredLimits;
    std::memset(&requiredLimits, 0, sizeof(requiredLimits));
    if (wgpuAdapterGetLimits(adapter, &requiredLimits) == WGPUStatus_Success)
    {
      requiredLimits.nextInChain = nullptr;

      // Adapters may report "unlimited" as UINT64_MAX, which collides with
      // WGPU_LIMIT_U64_UNDEFINED ("use the default") when passed back as a
      // required limit; substitute a large concrete value
      if (requiredLimits.maxBufferSize == UINT64_MAX)
        requiredLimits.maxBufferSize = 1ull << 34; // 16 GB

      devDesc.requiredLimits = &requiredLimits;

      if (isVerbose())
        std::cout << "  Limits  : maxBufferSize=" << requiredLimits.maxBufferSize
                  << ", maxStorageBufferBindingSize=" << requiredLimits.maxStorageBufferBindingSize
                  << std::endl;
    }

    WGPURequestDeviceCallbackInfo deviceCb{};
    deviceCb.callback = handle_request_device;
    deviceCb.userdata1 = &device;
  #if defined(__EMSCRIPTEN__)
    deviceCb.mode = WGPUCallbackMode_WaitAnyOnly;
    waitFuture(wgpuAdapterRequestDevice(adapter, &devDesc, deviceCb));
  #else
    deviceCb.mode = WGPUCallbackMode_AllowProcessEvents;
    wgpuAdapterRequestDevice(adapter, &devDesc, deviceCb);
    while (!device)
      wgpuInstanceProcessEvents(instance);
  #endif
    if (!device)
      throw std::runtime_error("failed to get WebGPU device");

    queue = wgpuDeviceGetQueue(device);

    if (isVerbose())
    {
      WGPULimits deviceLimits;
      std::memset(&deviceLimits, 0, sizeof(deviceLimits));
      if (wgpuDeviceGetLimits(device, &deviceLimits) == WGPUStatus_Success)
        std::cout << "  DevLim  : maxBufferSize=" << deviceLimits.maxBufferSize
                  << ", maxStorageBufferBindingSize=" << deviceLimits.maxStorageBufferBindingSize
                  << std::endl;
    }

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
    // There is no blocking device poll in the browser: wait on a queue
    // work-done future instead (awaits the promise via Asyncify)
    WGPUQueueWorkDoneCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void*, void*) {};
    waitFuture(wgpuQueueOnSubmittedWorkDone(queue, cb));
  #else
    wgpuDevicePoll(device, true, nullptr);
  #endif
  }

  void WebGPUDevice::waitFuture(WGPUFuture future)
  {
  #if defined(__EMSCRIPTEN__)
    WGPUFutureWaitInfo info{};
    info.future = future;
    wgpuInstanceWaitAny(instance, 1, &info, UINT64_MAX);
  #else
    // Native code uses blocking polls instead (see sync())
    (void)future;
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
