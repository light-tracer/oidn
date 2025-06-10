#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../devices/cpu/cpu_device.h"
#include "../devices/cpu/cpu_engine.h"
#include "../devices/cpu/cpu_pool.h"
#include "../core/tensor.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, Pool2x2)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t N=1,C=1,H=4,W=4;
  float src[N*C*H*W];
  for(size_t i=0;i<N*C*H*W;++i)
    src[i] = float(i+1);
  uint32_t OH = H/2, OW = W/2;
  float out[C*OH*OW];
  float ref[C*OH*OW];

  if (isCPUDeviceSupported())
  {
    auto cpuDev = newDevice(DeviceType::CPU);
    cpuDev.commit();
    auto cpuImpl = static_cast<CPUDevice*>(reinterpret_cast<Device*>(cpuDev.getHandle()));
    CPUEngine* cpuEng = static_cast<CPUEngine*>(cpuImpl->getEngine());
    const int blockC = cpuImpl->getTensorBlockC();

    TensorDesc srcDesc({int(C),int(H),int(W)},
                       {round_up(int(C),blockC),int(H),int(W)},
                       cpuImpl->getTensorLayout(), DataType::Float32);
    auto srcTensor = makeRef<HostTensor>(srcDesc);
    auto packChw = [&](float* dst)
    {
      for(int c=0;c<round_up(int(C),blockC);++c)
        for(int h=0;h<int(H);++h)
          for(int w=0;w<int(W);++w)
          {
            size_t idx=((size_t)(c/blockC)*H + h)*(W*blockC)+w*blockC+(c%blockC);
            if(c<int(C))
              dst[idx]=src[(size_t)c*H*W+h*W+w];
            else
              dst[idx]=0.f;
          }
    };
    packChw(static_cast<float*>(srcTensor->getPtr()));

    auto pool = cpuEng->newPool({srcDesc});
    pool->setSrc(srcTensor);
    auto dstDesc = pool->getDstDesc();
    auto dstTensor = makeRef<HostTensor>(dstDesc);
    pool->setDst(dstTensor);
    pool->submit(nullptr);
    cpuDev.sync();

    auto unpackChw = [&](const float* srcPacked)
    {
      for(int c=0;c<int(C);++c)
        for(int h=0;h<int(OH);++h)
          for(int w=0;w<int(OW);++w)
          {
            size_t idx=((size_t)(c/blockC)*OH + h)*(OW*blockC)+w*blockC+(c%blockC);
            ref[(size_t)c*OH*OW+h*OW+w]=srcPacked[idx];
          }
    };
    unpackChw(static_cast<float*>(dstTensor->getPtr()));
  }
  else
  {
    for(uint32_t y=0;y<OH;++y)
      for(uint32_t x=0;x<OW;++x)
      {
        float v0 = src[(y*2)*W + x*2];
        float v1 = src[(y*2)*W + x*2+1];
        float v2 = src[(y*2+1)*W + x*2];
        float v3 = src[(y*2+1)*W + x*2+1];
        float m = std::max(std::max(v0,v1), std::max(v2,v3));
        ref[y*OW+x] = m;
      }
  }

  float srcGPU[N*C*H*W];
  for(uint32_t h=0; h<H; ++h)
    for(uint32_t w=0; w<W; ++w)
      for(uint32_t c=0; c<C; ++c)
        srcGPU[(h*W + w)*C + c] = src[(size_t)c*H*W + h*W + w];

  auto srcBuf = dev.newBuffer(sizeof(srcGPU));
  srcBuf.write(0, sizeof(srcGPU), srcGPU);
  auto outBuf = dev.newBuffer(sizeof(out));

  TensorDesc srcDescGPU({int(C),int(H),int(W)}, TensorLayout::hwc, DataType::Float32);
  auto srcTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(srcBuf.getHandle())), srcDescGPU);

  auto pool = eng->newPool({srcDescGPU});
  pool->setSrc(srcTensorGPU);
  auto dstDescGPU = pool->getDstDesc();
  ASSERT_EQ(dstDescGPU.getH(), int(OH));
  ASSERT_EQ(dstDescGPU.getW(), int(OW));
  auto dstTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(outBuf.getHandle())), dstDescGPU);
  pool->setDst(dstTensorGPU);
  pool->submit(nullptr);
  dev.sync();
  float outGPU[C*OH*OW];
  outBuf.read(0, sizeof(outGPU), outGPU);
  for(uint32_t h=0; h<OH; ++h)
    for(uint32_t w=0; w<OW; ++w)
      for(uint32_t c=0; c<C; ++c)
        out[(size_t)c*OH*OW + h*OW + w] = outGPU[(h*OW + w)*C + c];

  for(size_t i=0;i<C*OH*OW;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}
