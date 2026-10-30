#include "dawn_compat.h"
// SPDX-License-Identifier: Apache-2.0
#include <OpenImageDenoise/oidn_webgpu.h>
#include "devices/webgpu/webgpu_device.h"
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstdio>

void testExternalUniforms(OIDNDevice oidn, WGPUDevice device, WGPUQueue queue, WGPUInstance instance)
{
  auto* engine = static_cast<oidn::WebGPUEngine*>(reinterpret_cast<oidn::WebGPUDevice*>(oidn)->getEngine());
  WGPUBufferDescriptor desc{};
  desc.size = 8; desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
  auto result = wgpuDeviceCreateBuffer(device, &desc);
  desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  auto readback = wgpuDeviceCreateBuffer(device, &desc);
  const auto& pipeline = engine->getPipeline("external-uniform-wrap", R"(
    struct Params { value:f32, index:u32, pad0:u32, pad1:u32 };
    @group(0) @binding(0) var<storage,read_write> result:array<f32>;
    @group(0) @binding(1) var<uniform> p:Params;
    @compute @workgroup_size(1) fn main(){result[p.index]+=p.value;}
  )", {WGPUBufferBindingType_Storage, WGPUBufferBindingType_Uniform});
  WGPUCommandBuffer commands[2]{};
  for (uint32_t frame = 0; frame < 2; ++frame)
  {
    auto encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    engine->beginExternalEncoding(encoder);
    for (uint32_t i = 1; i <= 1100; ++i)
    {
      struct Params { float value; uint32_t index, pad[2]; } p{float(i * (frame + 1)), frame, {0, 0}};
      const auto slot = engine->allocUniform(&p, sizeof(p));
      WGPUBindGroupEntry entries[2]{};
      entries[0].binding = 0; entries[0].buffer = result; entries[0].size = 8;
      entries[1].binding = 1; entries[1].buffer = slot.buffer; entries[1].offset = slot.offset; entries[1].size = sizeof(p);
      WGPUBindGroupDescriptor bg{};
      bg.layout = pipeline.bindGroupLayout; bg.entryCount = 2; bg.entries = entries;
      auto group = wgpuDeviceCreateBindGroup(device, &bg);
      auto pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
      wgpuComputePassEncoderSetPipeline(pass, pipeline.pipeline);
      wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
      wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
      wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass); wgpuBindGroupRelease(group);
    }
    engine->endExternalEncoding();
    if (frame == 1) wgpuCommandEncoderCopyBufferToBuffer(encoder, result, 0, readback, 0, 8);
    commands[frame] = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuCommandEncoderRelease(encoder);
  }
  wgpuQueueSubmit(queue, 2, commands);
  for (auto command : commands) wgpuCommandBufferRelease(command);
  struct Map { bool done = false, ok = false; } mapped;
  WGPUBufferMapCallbackInfo2 callback{};
  callback.mode = WGPUCallbackMode_AllowProcessEvents; callback.userdata1 = &mapped;
  callback.callback = [](WGPUMapAsyncStatus status, TestMessage, void* data, void*)
  { auto* result = static_cast<Map*>(data); result->ok = status == WGPUMapAsyncStatus_Success; result->done = true; };
  wgpuBufferMapAsync2(readback, WGPUMapMode_Read, 0, 8, callback);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (!mapped.done && std::chrono::steady_clock::now() < deadline)
  { wgpuInstanceProcessEvents(instance); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
  if (!mapped.ok) throw std::runtime_error("external uniform readback failed");
  const auto* values = static_cast<const float*>(wgpuBufferGetConstMappedRange(readback, 0, 8));
  if (!values || values[0] != 605550.f || values[1] != 1211100.f)
    throw std::runtime_error("external uniform pages overwritten before submission");
  wgpuBufferUnmap(readback); wgpuBufferRelease(readback); wgpuBufferRelease(result);
  std::puts("PASS: two unsubmitted encoders, 1100 distinct uniforms each");
}
