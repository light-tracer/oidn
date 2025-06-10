#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../devices/cpu/cpu_device.h"
#include "../devices/cpu/cpu_engine.h"
#include "../devices/cpu/cpu_autoexposure.h"
#include "../core/record.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, Autoexposure)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t H=2, W=2;
  float color[H*W*3] = {0.1f,0.2f,0.3f, 0.4f,0.5f,0.6f, 0.7f,0.8f,0.9f, 0.3f,0.2f,0.1f};
  float refVal = 1.f;

  if (isCPUDeviceSupported())
  {
    auto cpuDev = newDevice(DeviceType::CPU);
    cpuDev.commit();
    auto cpuImpl = static_cast<CPUDevice*>(reinterpret_cast<Device*>(cpuDev.getHandle()));
    CPUEngine* cpuEng = static_cast<CPUEngine*>(cpuImpl->getEngine());
    auto autoex = cpuEng->newAutoexposure(ImageDesc(Format::Float3,W,H));
    auto src = makeRef<Image>(color, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    auto buf = cpuEng->Engine::newBuffer(sizeof(float), Storage::Host);
    auto dst = makeRef<Record<float>>(buf,0);
    autoex->setSrc(src);
    autoex->setDst(dst);
    autoex->submit(nullptr);
    cpuDev.sync();
    refVal = *dst->getPtr();
  }
  else
  {
    refVal = 1.f; // fallback
  }

  auto autoex = eng->newAutoexposure(ImageDesc(Format::Float3,W,H));
  auto src = makeRef<Image>(color, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  auto buf = eng->Engine::newBuffer(sizeof(float), Storage::Host);
  auto dst = makeRef<Record<float>>(buf,0);
  autoex->setSrc(src);
  autoex->setDst(dst);
  autoex->submit(nullptr);
  dev.sync();
  float out = *dst->getPtr();

  ASSERT_NEAR(out, refVal, 1e-6f);
}

