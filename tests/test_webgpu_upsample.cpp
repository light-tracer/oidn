#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../devices/cpu/cpu_device.h"
#include "../devices/cpu/cpu_engine.h"
#include "../devices/cpu/cpu_upsample.h"
#include "../core/tensor.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, Upsample2x)
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
  uint32_t OH = H*2, OW = W*2;
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

    auto upsample = cpuEng->newUpsample({srcDesc});
    upsample->setSrc(srcTensor);
    auto dstDesc = upsample->getDstDesc();
    auto dstTensor = makeRef<HostTensor>(dstDesc);
    upsample->setDst(dstTensor);
    upsample->submit(nullptr);
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
        ref[y*OW+x] = src[(y/2)*W + (x/2)];
  }

  auto A = eng->newTensor(src, WebGPUTensorType::INPUT, N,C,H,W);
  auto O = eng->newTensor(out, WebGPUTensorType::OUTPUT, C,1,OH,OW);

  eng->upsample2x(A,O);
  eng->sync();

  for(size_t i=0;i<C*OH*OW;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}
