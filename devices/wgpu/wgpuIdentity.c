#include <webgpu/wgpu.h>
#include <webgpu/webgpu.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static void handle_request_adapter(WGPURequestAdapterStatus status,
                                   WGPUAdapter adapter, WGPUStringView message,
                                   void *userdata1, void *userdata2)
{
  (void)message; (void)userdata2;
  if (status == WGPURequestAdapterStatus_Success)
    *(WGPUAdapter*)userdata1 = adapter;
}

static void handle_request_device(WGPURequestDeviceStatus status,
                                  WGPUDevice device, WGPUStringView message,
                                  void *userdata1, void *userdata2)
{
  (void)message; (void)userdata2;
  if (status == WGPURequestDeviceStatus_Success)
    *(WGPUDevice*)userdata1 = device;
}

static void handle_buffer_map(WGPUMapAsyncStatus status,
                              WGPUStringView message,
                              void *userdata1, void *userdata2)
{
  (void)message; (void)userdata2;
  bool* done = userdata1;
  *done = (status == WGPUMapAsyncStatus_Success);
}

int main(void)
{
  wgpuSetLogLevel(WGPULogLevel_Warn);

  WGPUInstance instance = wgpuCreateInstance(NULL);
  assert(instance);

  WGPUAdapter adapter = NULL;
  WGPUFuture fut = wgpuInstanceRequestAdapter(instance, NULL,
      (WGPURequestAdapterCallbackInfo){ .callback = handle_request_adapter,
                                        .userdata1 = &adapter });
  WGPUFutureWaitInfo wi = { fut, false };
  wgpuInstanceWaitAny(instance, 1, &wi, UINT64_MAX);
  assert(adapter);

  WGPUDevice device = NULL;
  fut = wgpuAdapterRequestDevice(adapter, NULL,
        (WGPURequestDeviceCallbackInfo){ .callback = handle_request_device,
                                          .userdata1 = &device });
  wi.future = fut; wi.completed = false;
  wgpuInstanceWaitAny(instance, 1, &wi, UINT64_MAX);
  assert(device);

  WGPUQueue queue = wgpuDeviceGetQueue(device);
  assert(queue);

  const char* shader =
    "@group(0) @binding(0) var<storage, read> src : array<f32>;"
    "@group(0) @binding(1) var<storage, read_write> dst : array<f32>;"
    "@compute @workgroup_size(64)"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {"
    "  dst[gid.x] = src[gid.x];"
    "}";

  WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(device,
      &(WGPUShaderModuleDescriptor){
        .nextInChain = (const WGPUChainedStruct*)&(const WGPUShaderSourceWGSL){
          .chain = (WGPUChainedStruct){ .sType = WGPUSType_ShaderSourceWGSL },
          .code = { shader, strlen(shader) }
        }
      });

  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    { .binding = 0, .visibility = WGPUShaderStage_Compute,
      .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
    { .binding = 1, .visibility = WGPUShaderStage_Compute,
      .buffer = { .type = WGPUBufferBindingType_Storage } }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .entryCount = 2,
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

  const uint32_t num = 1024;
  size_t byte_size = num * sizeof(float);
  float data[1024];
  for (uint32_t i=0;i<num;i++) data[i] = 42.f;

  WGPUBuffer src = wgpuDeviceCreateBuffer(device,
      &(WGPUBufferDescriptor){ .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                               .size = byte_size, .mappedAtCreation = true });
  memcpy(wgpuBufferGetMappedRange(src,0,byte_size), data, byte_size);
  wgpuBufferUnmap(src);

  WGPUBuffer dst = wgpuDeviceCreateBuffer(device,
      &(WGPUBufferDescriptor){ .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapRead,
                               .size = byte_size, .mappedAtCreation = false });

  WGPUBindGroupEntry bg_entries[2] = {
    { .binding = 0, .buffer = src, .offset = 0, .size = byte_size },
    { .binding = 1, .buffer = dst, .offset = 0, .size = byte_size }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .layout = bgl,
    .entryCount = 2,
    .entries = bg_entries
  };
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bg_desc);

  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
  wgpuComputePassEncoderSetPipeline(pass, pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(pass, num, 1, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd);
  wgpuDevicePoll(device, true, NULL);

  bool mapped = false;
  wgpuBufferMapAsync(dst, WGPUMapMode_Read, 0, byte_size,
                     (WGPUBufferMapCallbackInfo){ .callback = handle_buffer_map, .userdata1=&mapped });
  while (!mapped)
    wgpuDevicePoll(device, true, NULL);

  float* out = (float*)wgpuBufferGetMappedRange(dst, 0, byte_size);
  bool passed = true;
  for (uint32_t i=0;i<num;i++)
    if (out[i] != 42.f) { passed = false; break; }
  wgpuBufferUnmap(dst);

  printf(passed ? "PASSED\n" : "FAILED\n");

  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);
  wgpuBindGroupRelease(bg);
  wgpuComputePipelineRelease(pipeline);
  wgpuPipelineLayoutRelease(pl);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuShaderModuleRelease(shader_module);
  wgpuBufferRelease(dst);
  wgpuBufferRelease(src);
  wgpuQueueRelease(queue);
  wgpuDeviceRelease(device);
  wgpuAdapterRelease(adapter);
  wgpuInstanceRelease(instance);
  return passed ? 0 : 1;
}
