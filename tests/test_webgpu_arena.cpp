#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_arena.h"
#include "../devices/webgpu/webgpu_buffer.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../devices/webgpu/webgpu_device.h"
#include "../common/platform.h"
#include <gtest/gtest.h>

using namespace oidn;

static void fillRandom(float* data, size_t count)
{
  for(size_t i=0;i<count;++i)
    data[i] = float(i%7)*0.2f + 1.0f;
}

TEST(WebGPU, Arena)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  constexpr size_t COUNT = 64;
  float a[COUNT];
  float b[COUNT];
  fillRandom(a, COUNT);
  fillRandom(b, COUNT);
  float ref[COUNT];
  for(size_t i=0;i<COUNT;++i)
    ref[i] = a[i] + b[i];

  const size_t bufSize = sizeof(a);
  const size_t heapSize = bufSize*2;
  auto arena = makeRef<WebGPUArena>(eng, heapSize);

  auto bufA = arena->newBuffer(bufSize, 0);
  auto bufB = arena->newBuffer(bufSize, bufSize);
  auto bufOutRef = dev.newBuffer(bufSize);

  bufA->write(0, bufSize, a);
  bufB->write(0, bufSize, b);

  WebGPUBuffer* wA = static_cast<WebGPUBuffer*>(bufA.get());
  WebGPUBuffer* wB = static_cast<WebGPUBuffer*>(bufB.get());
  WebGPUBuffer* wOut = static_cast<WebGPUBuffer*>(reinterpret_cast<Buffer*>(bufOutRef.getHandle()));

  WebGPUTensor tA{wA->getWGPUBuffer(), 0, 1,1,1,COUNT, WebGPUTensorType::Input};
  WebGPUTensor tB{wB->getWGPUBuffer(), bufSize, 1,1,1,COUNT, WebGPUTensorType::Input};
  WebGPUTensor tOut{wOut->getWGPUBuffer(), 0, 1,1,1,COUNT, WebGPUTensorType::Input};

  eng->add(tA,tB,tOut);
  dev.sync();

  float out[COUNT];
  bufOutRef.read(0, bufSize, out);
  for(size_t i=0;i<COUNT;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}

TEST(WebGPU, UniformArenaWrap)
{
  if (!isWebGPUDeviceSupported()) GTEST_SKIP();
  auto device = newWebGPUDevice();
  device.commit();
  auto* engine = getEngine(device);
  auto* native = static_cast<WebGPUDevice*>(engine->getDevice());
  auto result = engine->newBuffer(sizeof(float), Storage::Device);
  const auto& pipeline = engine->getPipeline("uniform-wrap-test", R"wgsl(
    struct Params { value:f32, pad0:u32, pad1:u32, pad2:u32 };
    @group(0) @binding(0) var<storage, read_write> result:array<f32>;
    @group(0) @binding(1) var<uniform> p:Params;
    @compute @workgroup_size(1) fn main() { result[0] += p.value; }
  )wgsl", {WGPUBufferBindingType_Storage, WGPUBufferBindingType_Uniform});

  constexpr uint32_t count = 1100; // the uniform arena contains 1024 slots
  for (uint32_t i = 1; i <= count; ++i)
  {
    struct Params { float value; uint32_t pad[3]; } params{float(i), {0,0,0}};
    const auto slot = engine->allocUniform(&params, sizeof(params));
    WGPUBindGroupEntry entries[2]{};
    entries[0].binding = 0;
    entries[0].buffer = static_cast<WebGPUBuffer*>(result.get())->getWGPUBuffer();
    entries[0].size = sizeof(float);
    entries[1].binding = 1; entries[1].buffer = slot.buffer;
    entries[1].offset = slot.offset; entries[1].size = sizeof(params);
    WGPUBindGroupDescriptor desc{};
    desc.layout = pipeline.bindGroupLayout; desc.entryCount = 2; desc.entries = entries;
    auto group = wgpuDeviceCreateBindGroup(native->device, &desc);
    auto pass = wgpuCommandEncoderBeginComputePass(engine->getEncoder(), nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuBindGroupRelease(group);
  }
  float sum = 0;
  result->read(0, sizeof(sum), &sum);
  ASSERT_EQ(sum, float(count*(count+1)/2));
}
