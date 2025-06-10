#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../devices/cpu/cpu_device.h"
#include "../devices/cpu/cpu_engine.h"
#include "../devices/cpu/cpu_image_copy.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, ImageCopy)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t H=2, W=2;
  float srcImg[H*W*3];
  for(size_t i=0;i<H*W*3;++i)
    srcImg[i] = float(i)*0.1f;

  float refImg[H*W*3];

  if (isCPUDeviceSupported())
  {
    auto cpuDev = newDevice(DeviceType::CPU);
    cpuDev.commit();
    auto cpuImpl = static_cast<CPUDevice*>(reinterpret_cast<Device*>(cpuDev.getHandle()));
    CPUEngine* cpuEng = static_cast<CPUEngine*>(cpuImpl->getEngine());
    auto copy = cpuEng->newImageCopy();
    auto src = makeRef<Image>(srcImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    auto dst = makeRef<Image>(refImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    copy->setSrc(src);
    copy->setDst(dst);
    copy->submit(nullptr);
    cpuDev.sync();
  }
  else
  {
    std::memcpy(refImg, srcImg, sizeof(refImg));
  }

  auto copy = eng->newImageCopy();
  auto src = makeRef<Image>(srcImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  float outImg[H*W*3];
  auto dst = makeRef<Image>(outImg, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  copy->setSrc(src);
  copy->setDst(dst);
  copy->submit(nullptr);
  dev.sync();

  for(size_t i=0;i<H*W*3;++i)
    ASSERT_NEAR(outImg[i], refImg[i], 1e-6f);
}

