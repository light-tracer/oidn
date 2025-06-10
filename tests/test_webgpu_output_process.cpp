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
    TensorDesc srcDesc({3,int(H),int(W)}, cpuImpl->getTensorLayout(), DataType::Float32);
    auto srcTensor = makeRef<HostTensor>(srcDesc, srcData);
    auto proc = cpuEng->newOutputProcess({srcDesc, std::make_shared<TransferFunction>(TransferFunction::Type::Linear), false, false});
    proc->setSrc(srcTensor);
    auto dstImg = makeRef<Image>(refImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    proc->setDst(dstImg);
    proc->submit(nullptr);
    cpuDev.sync();
  }
  else
  {
    for(size_t i=0;i<H*W*3;++i) refImg[i]=srcData[i];
  }

  TensorDesc srcDescGPU({3,int(H),int(W)}, TensorLayout::hwc, DataType::Float32);
  auto srcBuf = dev.newBuffer(sizeof(srcData));
  srcBuf.write(0, sizeof(srcData), srcData);
  auto srcTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(srcBuf.getHandle())), srcDescGPU);
  auto proc = eng->newOutputProcess({srcDescGPU, std::make_shared<TransferFunction>(TransferFunction::Type::Linear), false, false});
  proc->setSrc(srcTensorGPU);
  float outImg[H*W*3];
  auto dstImgGPU = makeRef<Image>(outImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  proc->setDst(dstImgGPU);
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
    TensorDesc srcDesc({3,int(H),int(W)}, cpuImpl->getTensorLayout(), DataType::Float32);
    auto srcTensor = makeRef<HostTensor>(srcDesc, srcData);
    auto proc = cpuEng->newOutputProcess({srcDesc, std::make_shared<TransferFunction>(TransferFunction::Type::SRGB), true, true});
    proc->setSrc(srcTensor);
    auto dstImg = makeRef<Image>(refImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    proc->setDst(dstImg);
    proc->submit(nullptr);
    cpuDev.sync();
  }

  TensorDesc srcDescGPU({3,int(H),int(W)}, TensorLayout::hwc, DataType::Float32);
  auto srcBuf = dev.newBuffer(sizeof(srcData));
  srcBuf.write(0, sizeof(srcData), srcData);
  auto srcTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(srcBuf.getHandle())), srcDescGPU);
  auto proc = eng->newOutputProcess({srcDescGPU, std::make_shared<TransferFunction>(TransferFunction::Type::SRGB), true, true});
  proc->setSrc(srcTensorGPU);
  float outImg[H*W*3];
  auto dstImgGPU = makeRef<Image>(outImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  proc->setDst(dstImgGPU);
  proc->submit(nullptr);
  dev.sync();

  if (isCPUDeviceSupported())
  {
    for(size_t i=0;i<H*W*3;++i)
      ASSERT_NEAR(outImg[i], refImg[i], 1e-4f);
  }
}

