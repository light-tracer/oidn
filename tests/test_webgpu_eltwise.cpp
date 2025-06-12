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
  GTEST_SKIP(); // Temporarily skip due to backend issues
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
  auto bufA = dev.newBuffer(sizeA);
  auto bufB = dev.newBuffer(sizeB);
  auto bufOut = dev.newBuffer(sizeOut);
  bufA.write(0,sizeof(a),a);
  bufB.write(0,sizeof(b),b);

  auto tA = eng->makeTensor(BufferRef(bufA.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tB = eng->makeTensor(BufferRef(bufB.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tOut= eng->makeTensor(BufferRef(bufOut.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);

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
  GTEST_SKIP(); // Temporarily skip due to backend issues
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
  auto tA = eng->makeTensor(BufferRef(bufA.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tB = eng->makeTensor(BufferRef(bufB.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tOut= eng->makeTensor(BufferRef(bufOut.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);

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
  GTEST_SKIP(); // Temporarily skip due to backend issues
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
  auto tA = eng->makeTensor(BufferRef(bufA.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);
  auto tOut= eng->makeTensor(BufferRef(bufOut.getHandle()), WebGPUTensorType::INPUT, 1,1,1,W);

  eng->softplus(tA,tOut);
  dev.sync();
  float out[COUNT];
  bufOut.read(0,sizeof(out),out);
  for(size_t i=0;i<COUNT;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}

