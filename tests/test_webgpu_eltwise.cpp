#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../core/tensor.h"
#include "../common/platform.h"
#include "../devices/webgpu/webgpu_arena.h"
#include <gtest/gtest.h>

using namespace oidn;

static void fillRandom(float* data, size_t count)
{
  for(size_t i=0;i<count;++i)
    data[i] = float(i%13) * 0.1f + 0.3f; // deterministic
}

TEST(WebGPU, EltwiseAdd)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();
  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t N=1,C=1,H=1,W=64; // 64 elements
  constexpr size_t COUNT = N*C*H*W;
  float a[COUNT];
  float b[COUNT];
  fillRandom(a, COUNT);
  fillRandom(b, COUNT);
  float ref[COUNT];
  for(size_t i=0;i<COUNT;++i)
    ref[i] = a[i] + b[i];

  const size_t sizeA = sizeof(a);
  const size_t sizeB = sizeof(b);
  const size_t sizeOut = sizeof(ref);
  size_t offA = 0;
  size_t offB = round_up(offA + sizeA, memoryAlignment);
  size_t offOut = round_up(offB + sizeB, memoryAlignment);
  size_t arenaSize = offOut + sizeOut;

  auto arena = makeRef<WebGPUArena>(eng, arenaSize);
  Ref<Buffer> bufAInt = eng->Engine::newBuffer(arena, sizeA, offA);
  Ref<Buffer> bufBInt = eng->Engine::newBuffer(arena, sizeB, offB);
  Ref<Buffer> bufOutInt = eng->Engine::newBuffer(arena, sizeOut, offOut);
  BufferRef bufA(reinterpret_cast<OIDNBuffer>(bufAInt.detach()));
  BufferRef bufB(reinterpret_cast<OIDNBuffer>(bufBInt.detach()));
  BufferRef bufOut(reinterpret_cast<OIDNBuffer>(bufOutInt.detach()));
  bufA.write(0,sizeof(a),a);
  bufB.write(0,sizeof(b),b);

  auto tA = eng->newTensor(BufferRef(bufA.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tB = eng->newTensor(BufferRef(bufB.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tOut= eng->newTensor(BufferRef(bufOut.getHandle()), WebGPUTensorType::OUTPUT, 1,1,1,W);

  eng->add(tA,tB,tOut);
  dev.sync();
  float out[COUNT];
  bufOut.read(0,sizeof(out),out);
  for(size_t i=0;i<COUNT;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}

TEST(WebGPU, EltwiseMul)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();
  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t N=1,C=1,H=1,W=64;
  constexpr size_t COUNT = N*C*H*W;
  float a[COUNT];
  float b[COUNT];
  fillRandom(a, COUNT);
  fillRandom(b, COUNT);
  float ref[COUNT];
  for(size_t i=0;i<COUNT;++i)
    ref[i]=a[i]*b[i];

  auto bufA=dev.newBuffer(sizeof(a)); bufA.write(0,sizeof(a),a);
  auto bufB=dev.newBuffer(sizeof(b)); bufB.write(0,sizeof(b),b);
  auto bufOut=dev.newBuffer(sizeof(ref));
  auto tA = eng->newTensor(BufferRef(bufA.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tB = eng->newTensor(BufferRef(bufB.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tOut= eng->newTensor(BufferRef(bufOut.getHandle()), WebGPUTensorType::OUTPUT, 1,1,1,W);

  eng->mul(tA,tB,tOut);
  dev.sync();
  float out[COUNT];
  bufOut.read(0,sizeof(out),out);
  for(size_t i=0;i<COUNT;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}

TEST(WebGPU, EltwiseSoftplus)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();
  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t N=1,C=1,H=1,W=64;
  constexpr size_t COUNT=N*C*H*W;
  float a[COUNT];
  fillRandom(a, COUNT);
  float ref[COUNT];
  for(size_t i=0;i<COUNT;++i)
    ref[i] = logf(1.f + expf(a[i]));

  auto bufA=dev.newBuffer(sizeof(a)); bufA.write(0,sizeof(a),a);
  auto bufOut=dev.newBuffer(sizeof(ref));
  auto tA = eng->newTensor(BufferRef(bufA.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tOut= eng->newTensor(BufferRef(bufOut.getHandle()), WebGPUTensorType::OUTPUT, 1,1,1,W);

  eng->softplus(tA,tOut);
  dev.sync();
  float out[COUNT];
  bufOut.read(0,sizeof(out),out);
  for(size_t i=0;i<COUNT;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}

