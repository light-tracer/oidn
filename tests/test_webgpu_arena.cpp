#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_arena.h"
#include "../devices/webgpu/webgpu_buffer.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

static void fillRandom(float* data, size_t count)
{
  for(size_t i=0;i<count;++i)
    data[i] = float(i%7)*0.2f + 1.0f;
}

TEST(WebGPU, Arena)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  constexpr size_t COUNT = 64;
  float a[COUNT];
  float b[COUNT];
  fillRandom(a, COUNT);
  fillRandom(b, COUNT);
  float ref[COUNT];
  for(size_t i=0;i<COUNT;++i)
    ref[i] = a[i] + b[i];

  const size_t bufSize = sizeof(a);
  const size_t heapSize = bufSize*2;
  auto arena = makeRef<WebGPUArena>(eng, heapSize);

  auto bufA = arena->newBuffer(bufSize, 0);
  auto bufB = arena->newBuffer(bufSize, bufSize);
  auto bufOutRef = dev.newBuffer(bufSize);

  bufA->write(0, bufSize, a);
  bufB->write(0, bufSize, b);

  WebGPUBuffer* wA = static_cast<WebGPUBuffer*>(bufA.get());
  WebGPUBuffer* wB = static_cast<WebGPUBuffer*>(bufB.get());
  WebGPUBuffer* wOut = static_cast<WebGPUBuffer*>(reinterpret_cast<Buffer*>(bufOutRef.getHandle()));

  WebGPUTensor tA{wA->getWGPUBuffer(), 0, 1,1,1,COUNT, WebGPUTensorType::INPUT};
  WebGPUTensor tB{wB->getWGPUBuffer(), bufSize, 1,1,1,COUNT, WebGPUTensorType::INPUT};
  WebGPUTensor tOut{wOut->getWGPUBuffer(), 0, 1,1,1,COUNT, WebGPUTensorType::INPUT};

  eng->add(tA,tB,tOut);
  dev.sync();

  float out[COUNT];
  bufOutRef.read(0, bufSize, out);
  for(size_t i=0;i<COUNT;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}

