#include "webgpu_buffer.h"
#include "webgpu_device.h"
#include "webgpu_engine.h"
#include "webgpu_heap.h"
#include <cstring>

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
    desc.size = roundByteSize4(byteSize);
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

    // Copy and map sizes must be multiples of 4 (the underlying buffers are
    // rounded up accordingly, so reading past the region end is in bounds)
    const size_t copySize = roundByteSize4(byteSize);

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    WGPUBuffer readback = dev->createBuffer(copySize,
                                 WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, buffer, this->byteOffset + byteOffset, readback, 0, copySize);
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
    cbInfo.callback = callback;
    cbInfo.userdata1 = &done;
  #if defined(__EMSCRIPTEN__)
    cbInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    dev->waitFuture(wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, copySize, cbInfo));
  #else
    cbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, copySize, cbInfo);
    while (!done)
      dev->sync();
  #endif
    if (!done)
      throw std::runtime_error("failed to map WebGPU readback buffer");
    // Note: must be the const variant; GetMappedRange fails on read-mapped
    // buffers in Dawn/emdawnwebgpu
    std::memcpy(dstHostPtr, wgpuBufferGetConstMappedRange(readback,0,copySize), byteSize);
    wgpuBufferUnmap(readback);
    wgpuBufferRelease(readback);
  }

  void WebGPUBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (srcHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "source host pointer is null");

    // Keep queue ordering: submit any pending compute work first
    engine->flush();

    // Copy sizes must be multiples of 4 (both buffers are rounded up
    // accordingly, so the extra bytes are in bounds)
    const size_t copySize = roundByteSize4(byteSize);

    auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
    // createBuffer rounds the buffer size up itself but copies only byteSize
    // bytes from the host pointer
    WGPUBuffer staging = dev->createBuffer(byteSize,
                                WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc, srcHostPtr);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, staging, 0, buffer, this->byteOffset + byteOffset, copySize);
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
