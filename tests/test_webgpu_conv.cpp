#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../devices/cpu/cpu_device.h"
#include "../devices/cpu/cpu_engine.h"
#include "../devices/cpu/cpu_conv.h"
#include "../core/tensor.h"
#include "../core/tensor_reorder.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, Conv2d)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t N=1,C=1,H=8,W=8;
  const uint32_t OC=1,IC=1,KH=3,KW=3;
  float src[N*C*H*W];
  float weight[OC*IC*KH*KW];
  float bias[OC];
  for(size_t i=0;i<N*C*H*W;++i) src[i]= (float)i*0.01f + 1.0f;
  for(size_t i=0;i<OC*IC*KH*KW;++i) weight[i]= (float)(i+1)*0.1f;
  bias[0]=0.5f;
  uint32_t OH=H-KH+1; uint32_t OW=W-KW+1;
  float out[OC*OH*OW];
  float ref[OC*OH*OW];

  // --- compute reference result on the CPU backend ---
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
    TensorDesc weightDesc({int(OC),int(IC),int(KH),int(KW)},
                          {round_up(int(OC),blockC),round_up(int(IC),blockC),int(KH),int(KW)},
                          cpuImpl->getWeightLayout(), DataType::Float32);
    TensorDesc biasDesc({int(OC)},
                        {round_up(int(OC),blockC)},
                        TensorLayout::x, DataType::Float32);

    auto srcTensor = makeRef<HostTensor>(srcDesc);
    half weightHalf[OC*IC*KH*KW];
    for(size_t i=0;i<OC*IC*KH*KW;++i)
      weightHalf[i] = half(weight[i]);
    auto weightSrc = makeRef<HostTensor>(TensorDesc({int(OC),int(IC),int(KH),int(KW)}, TensorLayout::oihw, DataType::Float16), weightHalf);
    auto weightTensor = makeRef<HostTensor>(weightDesc);
    half biasHalf[OC];
    for(size_t i=0;i<OC;++i)
      biasHalf[i] = half(bias[i]);
    auto biasSrc = makeRef<HostTensor>(TensorDesc({int(OC)}, TensorLayout::x, DataType::Float16), biasHalf);
    auto biasTensor = makeRef<HostTensor>(biasDesc);

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

    reorderWeight(*weightSrc, *weightTensor);
    reorderBias(*biasSrc, *biasTensor);

    auto convCPU = cpuEng->newConv({srcDesc, weightDesc, biasDesc, Activation::ReLU, PostOp::None, false});
    convCPU->setSrc(srcTensor);
    convCPU->setWeight(weightTensor);
    convCPU->setBias(biasTensor);
    auto dstDesc = convCPU->getDstDesc();
    auto dstTensor = makeRef<HostTensor>(dstDesc);
    convCPU->setDst(dstTensor);
    convCPU->submit(nullptr);
    cpuDev.sync();

    auto unpackChwFull = [&](const float* srcPacked, int HH, int WW, float* dst)
    {
      for(int c=0;c<int(OC);++c)
        for(int h=0;h<HH;++h)
          for(int w=0;w<WW;++w)
          {
            size_t idx=((size_t)(c/blockC)*HH + h)*(WW*blockC)+w*blockC+(c%blockC);
            dst[(size_t)c*HH*WW+h*WW+w]=srcPacked[idx];
          }
    };

    float tmp[OC*H*W];
    unpackChwFull(static_cast<float*>(dstTensor->getPtr()), H, W, tmp);
    for(int c=0;c<int(OC);++c)
      for(int h=0;h<int(OH);++h)
        for(int w=0;w<int(OW);++w)
          ref[(size_t)c*OH*OW+h*OW+w]=tmp[(size_t)c*H*W+(h+1)*W+(w+1)];
  }
  else
  {
    for(uint32_t y=0;y<OH;++y)
      for(uint32_t x=0;x<OW;++x)
      {
        float acc=0.f;
        for(uint32_t ky=0;ky<KH;++ky)
          for(uint32_t kx=0;kx<KW;++kx)
            acc+=src[(y+ky)*W+(x+kx)]*weight[ky*KW+kx];
        acc+=bias[0];
        if(acc<0.f) acc=0.f;
        ref[y*OW+x]=acc;
      }
  }

  float srcGPU[N*C*H*W];
  for(uint32_t h=0; h<H; ++h)
    for(uint32_t w=0; w<W; ++w)
      for(uint32_t c=0; c<C; ++c)
        srcGPU[(h*W + w)*C + c] = src[(size_t)c*H*W + h*W + w];

  auto srcBuf = dev.newBuffer(sizeof(srcGPU));
  srcBuf.write(0, sizeof(srcGPU), srcGPU);
  auto wBuf = dev.newBuffer(sizeof(weight));
  wBuf.write(0, sizeof(weight), weight);
  auto bBuf = dev.newBuffer(sizeof(bias));
  bBuf.write(0, sizeof(bias), bias);
  auto outBuf = dev.newBuffer(sizeof(out));

  TensorDesc srcDescGPU({int(C),int(H),int(W)}, TensorLayout::hwc, DataType::Float32);
  TensorDesc wDescGPU({int(OC),int(IC),int(KH),int(KW)}, TensorLayout::oihw, DataType::Float32);
  TensorDesc bDescGPU({int(OC)}, TensorLayout::x, DataType::Float32);

  auto srcTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(srcBuf.getHandle())), srcDescGPU);
  auto wTensorGPU   = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(wBuf.getHandle())), wDescGPU);
  auto bTensorGPU   = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(bBuf.getHandle())), bDescGPU);

  auto conv = eng->newConv({srcDescGPU, wDescGPU, bDescGPU, Activation::ReLU, PostOp::None, false});
  conv->setSrc(srcTensorGPU);
  conv->setWeight(wTensorGPU);
  conv->setBias(bTensorGPU);
  auto dstDescGPU = conv->getDstDesc();
  ASSERT_EQ(dstDescGPU.getH(), int(OH));
  ASSERT_EQ(dstDescGPU.getW(), int(OW));
  auto dstTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(outBuf.getHandle())), dstDescGPU);
  conv->setDst(dstTensorGPU);
  conv->submit(nullptr);
  dev.sync();

  float outGPU[OC*OH*OW];
  outBuf.read(0, sizeof(outGPU), outGPU);
  for(uint32_t h=0; h<OH; ++h)
    for(uint32_t w=0; w<OW; ++w)
      for(uint32_t c=0; c<OC; ++c)
        out[(size_t)c*OH*OW + h*OW + w] = outGPU[(h*OW + w)*OC + c];

  for(size_t i=0;i<OC*OH*OW;++i)
    ASSERT_NEAR(out[i], ref[i], 2e-4f);
}

