#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../devices/cpu/cpu_device.h"
#include "../devices/cpu/cpu_engine.h"
#include "../devices/cpu/cpu_input_process.h"
#include "../core/tensor.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, InputProcess)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t H=2, W=2;
  float color[H*W*3];
  for(size_t i=0;i<H*W*3;++i)
    color[i]= float(i)*0.1f; // within [0,1)

  float ref[3*H*W];

  if (isCPUDeviceSupported())
  {
    auto cpuDev = newDevice(DeviceType::CPU);
    cpuDev.commit();
    auto cpuImpl = static_cast<CPUDevice*>(reinterpret_cast<Device*>(cpuDev.getHandle()));
    CPUEngine* cpuEng = static_cast<CPUEngine*>(cpuImpl->getEngine());
    TensorDims dims{3,int(H),int(W)};
    auto tf = std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
    auto proc = cpuEng->newInputProcess({dims, tf, false, false});
    auto colorImg = makeRef<Image>(color, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    proc->setSrc(colorImg, nullptr, nullptr);
    auto dstDesc = proc->getDstDesc();
    auto dstTensor = makeRef<HostTensor>(dstDesc);
    proc->setDst(dstTensor);
    proc->submit(nullptr);
    cpuDev.sync();
    std::memcpy(ref, dstTensor->getPtr(), sizeof(ref));
  }
  else
  {
    for(size_t i=0;i<H*W*3;++i) ref[i]=color[i];
  }

  TensorDims dims{3,int(H),int(W)};
  auto tf = std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
  auto proc = eng->newInputProcess({dims, tf, false, false});
  auto colorImg = makeRef<Image>(color, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  proc->setSrc(colorImg, nullptr, nullptr);
  auto dstDescGPU = proc->getDstDesc();
  auto buf = dev.newBuffer(dstDescGPU.getByteSize());
  auto dstTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(buf.getHandle())), dstDescGPU);
  proc->setDst(dstTensorGPU);
  proc->submit(nullptr);
  dev.sync();
  float out[3*H*W];
  buf.read(0, sizeof(out), out);

  for(size_t i=0;i<3*H*W;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}

TEST(WebGPU, InputProcessAdvanced)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t H=2, W=2;
  float color[H*W*3];
  float albedo[H*W*3];
  float normal[H*W*3];
  for(size_t i=0;i<H*W*3;++i)
  {
    color[i]  = -0.5f + float(i)*0.5f;
    albedo[i] = float(i)*0.1f;
    normal[i] = -1.f + float(i)*0.2f;
  }

  float ref[9*H*W];

  if (isCPUDeviceSupported())
  {
    auto cpuDev = newDevice(DeviceType::CPU);
    cpuDev.commit();
    auto cpuImpl = static_cast<CPUDevice*>(reinterpret_cast<Device*>(cpuDev.getHandle()));
    CPUEngine* cpuEng = static_cast<CPUEngine*>(cpuImpl->getEngine());
    TensorDims dims{9,int(H),int(W)};
    auto tf = std::make_shared<TransferFunction>(TransferFunction::Type::SRGB);
    auto proc = cpuEng->newInputProcess({dims, tf, true, true});
    auto colorImg = makeRef<Image>(color, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    auto albedoImg = makeRef<Image>(albedo, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    auto normalImg = makeRef<Image>(normal, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
    proc->setSrc(colorImg, albedoImg, normalImg);
    auto dstDesc = proc->getDstDesc();
    auto dstTensor = makeRef<HostTensor>(dstDesc);
    proc->setDst(dstTensor);
    proc->submit(nullptr);
    cpuDev.sync();
    std::memcpy(ref, dstTensor->getPtr(), sizeof(ref));
  }
  else
  {
    for(size_t i=0;i<9*H*W;++i) ref[i]=0.f; // not used
  }

  TensorDims dims{9,int(H),int(W)};
  auto tf = std::make_shared<TransferFunction>(TransferFunction::Type::SRGB);
  auto proc = eng->newInputProcess({dims, tf, true, true});
  auto colorImg = makeRef<Image>(color, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  auto albedoImg = makeRef<Image>(albedo, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  auto normalImg = makeRef<Image>(normal, Format::Float3, W, H, 0, sizeof(float)*3, sizeof(float)*3*W);
  proc->setSrc(colorImg, albedoImg, normalImg);
  auto dstDescGPU = proc->getDstDesc();
  auto buf = dev.newBuffer(dstDescGPU.getByteSize());
  auto dstTensorGPU = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(buf.getHandle())), dstDescGPU);
  proc->setDst(dstTensorGPU);
  proc->submit(nullptr);
  dev.sync();
  float out[9*H*W];
  buf.read(0, sizeof(out), out);

  if (isCPUDeviceSupported())
  {
    for(size_t i=0;i<9*H*W;++i)
      ASSERT_NEAR(out[i], ref[i], 1e-4f);
  }
}

