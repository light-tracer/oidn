#include <webgpu/wgpu.h>
#include <webgpu/webgpu.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

static void handle_request_adapter(WGPURequestAdapterStatus status,
                                   WGPUAdapter adapter, WGPUStringView message,
                                   void *userdata1, void *userdata2)
{
  (void)message;
  bool* done = userdata2;
  if (status == WGPURequestAdapterStatus_Success)
    *(WGPUAdapter*)userdata1 = adapter;
  *done = true;
}

static void handle_request_device(WGPURequestDeviceStatus status,
                                  WGPUDevice device, WGPUStringView message,
                                  void *userdata1, void *userdata2)
{
  (void)message;
  bool* done = userdata2;
  if (status == WGPURequestDeviceStatus_Success)
    *(WGPUDevice*)userdata1 = device;
  *done = true;
}

static void handle_buffer_map(WGPUMapAsyncStatus status,
                              WGPUStringView message,
                              void *userdata1, void *userdata2)
{
  (void)message; (void)userdata2;
  bool* done = userdata1;
  *done = (status == WGPUMapAsyncStatus_Success);
}

#define IN_W 8u
#define IN_H 8u
#define K_W 3u
#define K_H 3u
#define OUT_W (IN_W - K_W + 1u)
#define OUT_H (IN_H - K_H + 1u)

static void conv2d_cpu(const float* src, const float* w, float bias, float* dst)
{
  for (unsigned y = 0; y < OUT_H; ++y)
  {
    for (unsigned x = 0; x < OUT_W; ++x)
    {
      float acc = 0.f;
      for (unsigned ky = 0; ky < K_H; ++ky)
        for (unsigned kx = 0; kx < K_W; ++kx)
          acc += src[(y+ky)*IN_W + (x+kx)] * w[ky*K_W + kx];
      acc += bias;
      if (acc < 0.f) acc = 0.f;
      dst[y*OUT_W + x] = acc;
    }
  }
}

int main(void)
{
  wgpuSetLogLevel(WGPULogLevel_Warn);

  WGPUInstance instance = wgpuCreateInstance(NULL);
  assert(instance);

  WGPUAdapter adapter = NULL;
  bool adapter_done = false;
  wgpuInstanceRequestAdapter(instance, NULL,
      (WGPURequestAdapterCallbackInfo){ .callback = handle_request_adapter,
                                        .userdata1 = &adapter,
                                        .userdata2 = &adapter_done });
  while (!adapter_done)
    wgpuInstanceProcessEvents(instance);
  assert(adapter);

  WGPUDevice device = NULL;
  bool device_done = false;
  wgpuAdapterRequestDevice(adapter, NULL,
        (WGPURequestDeviceCallbackInfo){ .callback = handle_request_device,
                                          .userdata1 = &device,
                                          .userdata2 = &device_done });
  while (!device_done)
    wgpuInstanceProcessEvents(instance);
  assert(device);

  WGPUQueue queue = wgpuDeviceGetQueue(device);
  assert(queue);

  const char* shader =
    "const IN_W:u32=8u;const IN_H:u32=8u;const K_W:u32=3u;const K_H:u32=3u;"\
    "const OUT_W:u32=6u;const OUT_H:u32=6u;"\
    "@group(0)@binding(0)var<storage,read> src:array<f32>;"\
    "@group(0)@binding(1)var<storage,read> weight:array<f32>;"\
    "@group(0)@binding(2)var<storage,read> bias:array<f32>;"\
    "@group(0)@binding(3)var<storage,read_write> dst:array<f32>;"\
    "@compute @workgroup_size(8,8)"\
    "fn main(@builtin(global_invocation_id) gid:vec3<u32>){"\
    "if(gid.x>=OUT_W||gid.y>=OUT_H){return;}"\
    "var acc:f32=0.0;"\
    "for(var ky:u32=0u;ky<K_H;ky++){for(var kx:u32=0u;kx<K_W;kx++){"\
    "let ix=gid.x+kx;let iy=gid.y+ky;"\
    "acc=acc+src[(iy*IN_W+ix)]*weight[(ky*K_W+kx)];}}"\
    "acc=acc+bias[0];"\
    "if(acc<0.0){acc=0.0;}"\
    "dst[gid.y*OUT_W+gid.x]=acc;"\
    "}";

  WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(device,
      &(WGPUShaderModuleDescriptor){
        .nextInChain = (const WGPUChainedStruct*)&(const WGPUShaderSourceWGSL){
          .chain = (WGPUChainedStruct){ .sType = WGPUSType_ShaderSourceWGSL },
          .code = { shader, strlen(shader) }
        }
      });

  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    { .binding = 0, .visibility = WGPUShaderStage_Compute,
      .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
    { .binding = 1, .visibility = WGPUShaderStage_Compute,
      .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
    { .binding = 2, .visibility = WGPUShaderStage_Compute,
      .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
    { .binding = 3, .visibility = WGPUShaderStage_Compute,
      .buffer = { .type = WGPUBufferBindingType_Storage } }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .entryCount = 4,
    .entries = bgl_entries
  };
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts = &bgl
  };
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUComputePipelineDescriptor cp_desc = {
    .layout = pl,
    .compute = { .module = shader_module, .entryPoint = { "main", 4 } }
  };
  WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &cp_desc);

  const uint32_t src_count = IN_W * IN_H;
  const uint32_t weight_count = K_W * K_H;
  const uint32_t dst_count = OUT_W * OUT_H;
  const size_t src_bytes = src_count * sizeof(float);
  const size_t weight_bytes = weight_count * sizeof(float);
  const size_t bias_bytes = sizeof(float);
  const size_t dst_bytes = dst_count * sizeof(float);

  float src[src_count];
  float weight[weight_count];
  float bias[1] = {0.5f};
  for (uint32_t i = 0; i < src_count; ++i) src[i] = (float)i * 0.01f + 1.0f;
  for (uint32_t i = 0; i < weight_count; ++i) weight[i] = (float)(i+1) * 0.1f;

  float ref[dst_count];
  conv2d_cpu(src, weight, bias[0], ref);

  WGPUBuffer src_buf = wgpuDeviceCreateBuffer(device,
      &(WGPUBufferDescriptor){ .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                               .size = src_bytes, .mappedAtCreation = true });
  memcpy(wgpuBufferGetMappedRange(src_buf,0,src_bytes), src, src_bytes);
  wgpuBufferUnmap(src_buf);

  WGPUBuffer weight_buf = wgpuDeviceCreateBuffer(device,
      &(WGPUBufferDescriptor){ .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                               .size = weight_bytes, .mappedAtCreation = true });
  memcpy(wgpuBufferGetMappedRange(weight_buf,0,weight_bytes), weight, weight_bytes);
  wgpuBufferUnmap(weight_buf);

  WGPUBuffer bias_buf = wgpuDeviceCreateBuffer(device,
      &(WGPUBufferDescriptor){ .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                               .size = bias_bytes, .mappedAtCreation = true });
  memcpy(wgpuBufferGetMappedRange(bias_buf,0,bias_bytes), bias, bias_bytes);
  wgpuBufferUnmap(bias_buf);

  WGPUBuffer dst_buf = wgpuDeviceCreateBuffer(device,
      &(WGPUBufferDescriptor){ .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                               .size = dst_bytes, .mappedAtCreation = false });

  WGPUBuffer readback = wgpuDeviceCreateBuffer(device,
      &(WGPUBufferDescriptor){ .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
                               .size = dst_bytes, .mappedAtCreation = false });

  WGPUBindGroupEntry bg_entries[4] = {
    { .binding = 0, .buffer = src_buf, .offset = 0, .size = src_bytes },
    { .binding = 1, .buffer = weight_buf, .offset = 0, .size = weight_bytes },
    { .binding = 2, .buffer = bias_buf, .offset = 0, .size = bias_bytes },
    { .binding = 3, .buffer = dst_buf, .offset = 0, .size = dst_bytes }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .layout = bgl,
    .entryCount = 4,
    .entries = bg_entries
  };
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bg_desc);

  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
  wgpuComputePassEncoderSetPipeline(pass, pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(pass, OUT_W, OUT_H, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  wgpuCommandEncoderCopyBufferToBuffer(enc, dst_buf, 0, readback, 0, dst_bytes);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd);
  wgpuDevicePoll(device, true, NULL);

  bool mapped = false;
  wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, dst_bytes,
                     (WGPUBufferMapCallbackInfo){ .callback = handle_buffer_map, .userdata1 = &mapped });
  while (!mapped)
    wgpuDevicePoll(device, true, NULL);

  float* out = (float*)wgpuBufferGetMappedRange(readback, 0, dst_bytes);
  bool passed = true;
  for (uint32_t i=0;i<dst_count;i++)
    if (fabsf(out[i] - ref[i]) > 1e-5f) { passed = false; break; }
  wgpuBufferUnmap(readback);

  printf(passed ? "PASSED\n" : "FAILED\n");

  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);
  wgpuBindGroupRelease(bg);
  wgpuComputePipelineRelease(pipeline);
  wgpuPipelineLayoutRelease(pl);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuShaderModuleRelease(shader_module);
  wgpuBufferRelease(readback);
  wgpuBufferRelease(dst_buf);
  wgpuBufferRelease(bias_buf);
  wgpuBufferRelease(weight_buf);
  wgpuBufferRelease(src_buf);
  wgpuQueueRelease(queue);
  wgpuDeviceRelease(device);
  wgpuAdapterRelease(adapter);
  wgpuInstanceRelease(instance);
  return passed ? 0 : 1;
}

