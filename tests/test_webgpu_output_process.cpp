#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../devices/cpu/cpu_device.h"
#include "../devices/cpu/cpu_engine.h"
#include "../devices/cpu/cpu_output_process.h"
#include "../core/tensor.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, OutputProcess)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t H=2, W=2;
  float srcData[H*W*3];
  for(size_t i=0;i<H*W*3;++i)
    srcData[i] = float(i)*0.1f;

  float refImg[H*W*3];

  if (isCPUDeviceSupported())
  {
    auto cpuDev = newDevice(DeviceType::CPU);
    cpuDev.commit();
    auto cpuImpl = static_cast<CPUDevice*>(reinterpret_cast<Device*>(cpuDev.getHandle()));
    CPUEngine* cpuEng = static_cast<CPUEngine*>(cpuImpl->getEngine());
    TensorDesc srcDesc({3,int(H),int(W)},
                       {round_up(3,cpuImpl->getTensorBlockC()),int(H),int(W)},
                       cpuImpl->getTensorLayout(), DataType::Float32);
    auto srcTensor = makeRef<HostTensor>(srcDesc);
    auto pack = [&](const float* src, float* dst)
    {
      const int blockC = cpuImpl->getTensorBlockC();
      for(int c=0;c<round_up(3,blockC);++c)
        for(int h=0;h<int(H);++h)
          for(int w=0;w<int(W);++w)
          {
            size_t idx=((size_t)(c/blockC)*H + h)*(W*blockC)+w*blockC+(c%blockC);
            if(c<3)
              dst[idx]=src[(size_t)c*H*W+h*W+w];
            else
              dst[idx]=0.f;
          }
    };
    pack(srcData, static_cast<float*>(srcTensor->getPtr()));
    auto proc = cpuEng->newOutputProcess({srcDesc, std::make_shared<TransferFunction>(TransferFunction::Type::Linear), false, false});
    proc->setSrc(srcTensor);
    auto dstImg = makeRef<Image>(refImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    proc->setDst(dstImg);
    proc->setTile(0, 0, 0, 0, H, W);
    proc->submit(nullptr);
    cpuDev.sync();
  }
  else
  {
    for(size_t i=0;i<H*W*3;++i) refImg[i]=srcData[i];
  }

  TensorDesc srcDescGPU({3,int(H),int(W)}, TensorLayout::hwc, DataType::Float32);
  float srcDataGPU[H*W*3];
  for(uint32_t h=0; h<H; ++h)
    for(uint32_t w=0; w<W; ++w)
      for(uint32_t c=0; c<3; ++c)
        srcDataGPU[(h*W + w)*3 + c] = srcData[(size_t)c*H*W + h*W + w];
  auto srcBuf = dev.newBuffer(sizeof(srcDataGPU));
  srcBuf.write(0, sizeof(srcDataGPU), srcDataGPU);
  auto srcTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(srcBuf.getHandle())), srcDescGPU);
  auto proc = eng->newOutputProcess({srcDescGPU, std::make_shared<TransferFunction>(TransferFunction::Type::Linear), false, false});
  proc->setSrc(srcTensorGPU);
  float outImg[H*W*3];
  auto dstImgGPU = makeRef<Image>(outImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  proc->setDst(dstImgGPU);
  proc->setTile(0, 0, 0, 0, H, W);
  proc->submit(nullptr);
  dev.sync();

  for(size_t i=0;i<H*W*3;++i)
    ASSERT_NEAR(outImg[i], refImg[i], 1e-6f);
}

TEST(WebGPU, OutputProcessAdvanced)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t H=2, W=2;
  float srcData[H*W*3];
  for(size_t i=0;i<H*W*3;++i)
    srcData[i] = -0.5f + float(i)*0.5f;

  float refImg[H*W*3];

  if (isCPUDeviceSupported())
  {
    auto cpuDev = newDevice(DeviceType::CPU);
    cpuDev.commit();
    auto cpuImpl = static_cast<CPUDevice*>(reinterpret_cast<Device*>(cpuDev.getHandle()));
    CPUEngine* cpuEng = static_cast<CPUEngine*>(cpuImpl->getEngine());
    TensorDesc srcDesc({3,int(H),int(W)},
                       {round_up(3,cpuImpl->getTensorBlockC()),int(H),int(W)},
                       cpuImpl->getTensorLayout(), DataType::Float32);
    auto srcTensor = makeRef<HostTensor>(srcDesc);
    auto pack = [&](const float* src, float* dst)
    {
      const int blockC = cpuImpl->getTensorBlockC();
      for(int c=0;c<round_up(3,blockC);++c)
        for(int h=0;h<int(H);++h)
          for(int w=0;w<int(W);++w)
          {
            size_t idx=((size_t)(c/blockC)*H + h)*(W*blockC)+w*blockC+(c%blockC);
            if(c<3)
              dst[idx]=src[(size_t)c*H*W+h*W+w];
            else
              dst[idx]=0.f;
          }
    };
    pack(srcData, static_cast<float*>(srcTensor->getPtr()));
    auto proc = cpuEng->newOutputProcess({srcDesc, std::make_shared<TransferFunction>(TransferFunction::Type::SRGB), true, true});
    proc->setSrc(srcTensor);
    auto dstImg = makeRef<Image>(refImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    proc->setDst(dstImg);
    proc->setTile(0, 0, 0, 0, H, W);
    proc->submit(nullptr);
    cpuDev.sync();
  }

  TensorDesc srcDescGPU({3,int(H),int(W)}, TensorLayout::hwc, DataType::Float32);
  float srcDataGPU[H*W*3];
  for(uint32_t h=0; h<H; ++h)
    for(uint32_t w=0; w<W; ++w)
      for(uint32_t c=0; c<3; ++c)
        srcDataGPU[(h*W + w)*3 + c] = srcData[(size_t)c*H*W + h*W + w];
  auto srcBuf = dev.newBuffer(sizeof(srcDataGPU));
  srcBuf.write(0, sizeof(srcDataGPU), srcDataGPU);
  auto srcTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(srcBuf.getHandle())), srcDescGPU);
  auto proc = eng->newOutputProcess({srcDescGPU, std::make_shared<TransferFunction>(TransferFunction::Type::SRGB), true, true});
  proc->setSrc(srcTensorGPU);
  float outImg[H*W*3];
  auto dstImgGPU = makeRef<Image>(outImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  proc->setDst(dstImgGPU);
  proc->setTile(0, 0, 0, 0, H, W);
  proc->submit(nullptr);
  dev.sync();

  if (isCPUDeviceSupported())
  {
    for(size_t i=0;i<H*W*3;++i)
      ASSERT_NEAR(outImg[i], refImg[i], 1e-4f);
  }
}

