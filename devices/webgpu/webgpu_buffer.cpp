#include "webgpu_buffer.h"
#include "webgpu_device.h"
#include "webgpu_engine.h"
#include "webgpu_heap.h"
#include <cstring>
#if defined(__EMSCRIPTEN__)
  #include <emscripten.h>
#endif

OIDN_NAMESPACE_BEGIN

  WebGPUBuffer::WebGPUBuffer(WebGPUEngine* engine, size_t byteSize)
    : engine(engine), buffer(nullptr), byteSize(byteSize), shared(false)
  {
    init();
  }

  WebGPUBuffer::WebGPUBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
    : Buffer(arena, byteOffset),
      engine(dynamic_cast<WebGPUEngine*>(arena->getEngine())),
      buffer(nullptr),
      byteSize(byteSize),
      shared(true)
  {
    if (!engine)
      throw Exception(Error::InvalidArgument, "buffer is incompatible with arena");

    const auto byteSizeAndAlignment = engine->getBufferByteSizeAndAlignment(byteSize, Storage::Device);
    if (byteOffset % byteSizeAndAlignment.alignment != 0)
      throw Exception(Error::InvalidArgument, "buffer offset is unaligned");
    if (byteOffset + byteSizeAndAlignment.size > arena->getByteSize())
      throw Exception(Error::InvalidArgument, "arena region is out of bounds");

    WebGPUHeap* heap = static_cast<WebGPUHeap*>(arena->getHeap());
    buffer = heap->getWGPUBuffer();
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
    if (!shared && buffer)
      wgpuBufferRelease(buffer);
    buffer = nullptr;
  }

  void WebGPUBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (dstHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "destination host pointer is null");

    // Make sure any pending compute work producing this buffer is submitted
    engine->flush();

  #if defined(__EMSCRIPTEN__) && defined(OIDN_WEBGPU_READ_TRACE)
    const double rt0 = emscripten_get_now();
  #endif

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    WGPUBuffer readback = dev->createBuffer(byteSize,
                                 WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

  #if defined(__EMSCRIPTEN__) && defined(OIDN_WEBGPU_READ_TRACE)
    const double rt1 = emscripten_get_now();
  #endif

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, buffer, this->byteOffset + byteOffset, readback, 0, byteSize);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    dev->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

  #if defined(__EMSCRIPTEN__) && defined(OIDN_WEBGPU_READ_TRACE)
    const double rt2 = emscripten_get_now();
  #endif

    bool done = false;
    auto callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*)
    {
      bool* d = static_cast<bool*>(userdata1);
      *d = (status == WGPUMapAsyncStatus_Success);
    };
    WGPUBufferMapCallbackInfo cbInfo{};
    cbInfo.callback = callback;
    cbInfo.userdata1 = &done;
  #if defined(__EMSCRIPTEN__)
    cbInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    dev->waitFuture(wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, byteSize, cbInfo));
  #else
    cbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, byteSize, cbInfo);
    while (!done)
      dev->sync();
  #endif
    if (!done)
      throw std::runtime_error("failed to map WebGPU readback buffer");

  #if defined(__EMSCRIPTEN__) && defined(OIDN_WEBGPU_READ_TRACE)
    const double rt3 = emscripten_get_now();
  #endif

    // Note: must be the const variant; GetMappedRange fails on read-mapped
    // buffers in Dawn/emdawnwebgpu
    std::memcpy(dstHostPtr, wgpuBufferGetConstMappedRange(readback,0,byteSize), byteSize);
    wgpuBufferUnmap(readback);
    wgpuBufferRelease(readback);

  #if defined(__EMSCRIPTEN__) && defined(OIDN_WEBGPU_READ_TRACE)
    const double rt4 = emscripten_get_now();
    printf("oidn read: alloc %.1f, submit %.1f, mapWait %.1f, copy %.1f ms (%zu bytes)\n",
           rt1 - rt0, rt2 - rt1, rt3 - rt2, rt4 - rt3, byteSize);
  #endif
  }

  void WebGPUBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (srcHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "source host pointer is null");

    // Keep queue ordering: submit any pending compute work first
    engine->flush();

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    WGPUBuffer staging = dev->createBuffer(byteSize,
                                WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc, srcHostPtr);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, staging, 0, buffer, this->byteOffset + byteOffset, byteSize);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    dev->submit(cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBufferRelease(staging);
  }

  void WebGPUBuffer::postRealloc()
  {
    if (arena)
    {
      WebGPUHeap* heap = static_cast<WebGPUHeap*>(arena->getHeap());
      buffer = heap->getWGPUBuffer();
    }

    Buffer::postRealloc();
  }

OIDN_NAMESPACE_END
