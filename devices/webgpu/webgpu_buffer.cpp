#include "webgpu_buffer.h"
#include "webgpu_device.h"
#include "webgpu_engine.h"
#include <cstring>

OIDN_NAMESPACE_BEGIN

  WebGPUBuffer::WebGPUBuffer(WebGPUEngine* engine, size_t byteSize)
    : engine(engine), buffer(nullptr), byteSize(byteSize)
  {
    init();
  }

  WebGPUBuffer::~WebGPUBuffer()
  {
    free();
  }

  void WebGPUBuffer::init()
  {
    if (byteSize == 0)
      return;
    WGPUBufferDescriptor desc{};
    desc.size = byteSize;
    desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                 WGPUBufferUsage_CopySrc;
    desc.mappedAtCreation = false;
    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    buffer = wgpuDeviceCreateBuffer(dev->device, &desc);
    if (!buffer)
      throw std::runtime_error("failed to create WebGPU buffer");
  }

  void WebGPUBuffer::free()
  {
    if (buffer)
      wgpuBufferRelease(buffer);
    buffer = nullptr;
  }

  void WebGPUBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (dstHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "destination host pointer is null");

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    WGPUBuffer readback = dev->createBuffer(byteSize,
                                 WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, buffer, byteOffset, readback, 0, byteSize);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    dev->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    bool done = false;
    auto callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*)
    {
      bool* d = static_cast<bool*>(userdata1);
      *d = (status == WGPUMapAsyncStatus_Success);
    };
    WGPUBufferMapCallbackInfo cbInfo{};
    cbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    cbInfo.callback = callback;
    cbInfo.userdata1 = &done;
    wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, byteSize, cbInfo);
    while (!done)
      dev->sync();
    std::memcpy(dstHostPtr, wgpuBufferGetMappedRange(readback,0,byteSize), byteSize);
    wgpuBufferUnmap(readback);
    wgpuBufferRelease(readback);
  }

  void WebGPUBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (srcHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "source host pointer is null");

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    WGPUBuffer staging = dev->createBuffer(byteSize,
                                WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc, srcHostPtr);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, staging, 0, buffer, byteOffset, byteSize);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    dev->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBufferRelease(staging);
  }

OIDN_NAMESPACE_END
