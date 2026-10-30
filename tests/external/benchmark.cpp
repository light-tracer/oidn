// SPDX-License-Identifier: Apache-2.0
// Uses the application's Dawn, not wgpu-native. Only this test reads back pixels.
#include <OpenImageDenoise/oidn_webgpu.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

static int validationErrors = 0;

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static void check(OIDNDevice device)
{
  const char* message = nullptr;
  if (oidnGetDeviceError(device, &message) != OIDN_ERROR_NONE)
    throw std::runtime_error(message ? message : "OIDN error");
}
static void wait(WGPUInstance instance, const bool& done)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
  while (!done && std::chrono::steady_clock::now() < deadline)
  {
    wgpuInstanceProcessEvents(instance);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(done, "Dawn callback timeout");
}
static std::vector<char> load(const char* path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(bool(file), path);
  std::vector<char> data(size_t(file.tellg()));
  file.seekg(0);
  file.read(data.data(), data.size());
  require(bool(file), "file read failed");
  return data;
}
static WGPUBuffer buffer(WGPUDevice device, size_t size, WGPUBufferUsage usage)
{
  WGPUBufferDescriptor desc{};
  desc.size = size;
  desc.usage = usage;
  return wgpuDeviceCreateBuffer(device, &desc);
}

#include "devices/webgpu/webgpu_device.h"
int main(int argc, char** argv) try
{
  require(argc == 8, "Usage: oidn_gpu_bench weights width height f16|f32 mode budget repeats");
  auto weights = load(argv[1]);
  const size_t width = std::atoi(argv[2]), height = std::atoi(argv[3]);
  WGPUInstance instance = wgpuCreateInstance(nullptr);
  WGPUAdapter adapter = nullptr;
  bool done = false;
  WGPURequestAdapterCallbackInfo2 adapterCB{};
  adapterCB.mode = WGPUCallbackMode_AllowProcessEvents;
  adapterCB.userdata1 = &adapter;
  adapterCB.userdata2 = &done;
  adapterCB.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* out, void* done)
  {
    if (status == WGPURequestAdapterStatus_Success) *static_cast<WGPUAdapter*>(out) = adapter;
    else std::fprintf(stderr, "Adapter: %s\n", message);
    *static_cast<bool*>(done) = true;
  };
  WGPURequestAdapterOptions adapterOptions{};
  adapterOptions.powerPreference = WGPUPowerPreference_HighPerformance;
  const char* dxc = "use_dxc";
  WGPUDawnTogglesDescriptor toggles{};
  toggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
  if (std::getenv("OIDN_TEST_DXC")) {
    toggles.enabledToggleCount = 1; toggles.enabledToggles = &dxc;
    adapterOptions.nextInChain = &toggles.chain;
  }
  if (std::getenv("OIDN_TEST_VULKAN")) adapterOptions.backendType = WGPUBackendType_Vulkan;
  if (std::getenv("OIDN_TEST_D3D12")) adapterOptions.backendType = WGPUBackendType_D3D12;
  wgpuInstanceRequestAdapter2(instance, &adapterOptions, adapterCB);
  wait(instance, done);
  require(adapter, "no Dawn adapter");
  WGPUAdapterInfo info{};
  wgpuAdapterGetInfo(adapter, &info);
  std::printf("adapter %s %s backend=%d\n", info.device, info.description, int(info.backendType));
  wgpuAdapterInfoFreeMembers(info);

  WGPUDevice device = nullptr;
  done = false;
  WGPUDeviceDescriptor deviceDesc{};
  std::printf("adapter shader-f16=%d timestamp=%d\n", wgpuAdapterHasFeature(adapter, WGPUFeatureName_ShaderF16), wgpuAdapterHasFeature(adapter, WGPUFeatureName_TimestampQuery));
  WGPUFeatureName features[] = {WGPUFeatureName_TimestampQuery, WGPUFeatureName_ShaderF16};
  WGPUFeatureName feature = WGPUFeatureName_ShaderF16;
  deviceDesc.requiredFeatures = features;
  deviceDesc.requiredFeatureCount = 1;
  if (std::strcmp(argv[4], "f16") == 0)
  {
    require(wgpuAdapterHasFeature(adapter, feature), "adapter has no shader-f16");
    deviceDesc.requiredFeatureCount = 2;
  }
  WGPUSupportedLimits supported{};
  wgpuAdapterGetLimits(adapter, &supported);
  WGPURequiredLimits limits{};
  limits.limits = supported.limits;
  deviceDesc.requiredLimits = &limits;
  deviceDesc.uncapturedErrorCallbackInfo2.callback = [](const WGPUDevice*, WGPUErrorType, const char* message, void*, void*)
  {
    ++validationErrors;
    std::fprintf(stderr, "Dawn validation: %s\n", message);
  };
  WGPURequestDeviceCallbackInfo2 deviceCB{};
  deviceCB.mode = WGPUCallbackMode_AllowProcessEvents;
  deviceCB.userdata1 = &device;
  deviceCB.userdata2 = &done;
  deviceCB.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, const char* message, void* out, void* done)
  {
    if (status == WGPURequestDeviceStatus_Success) *static_cast<WGPUDevice*>(out) = device;
    else std::fprintf(stderr, "Device: %s\n", message);
    *static_cast<bool*>(done) = true;
  };
  wgpuAdapterRequestDevice2(adapter, &deviceDesc, deviceCB);
  wait(instance, done);
  require(device, "Dawn device creation failed");
  WGPUQueue queue = wgpuDeviceGetQueue(device);
  OIDNWebGPUContextDesc context{sizeof(context), device, queue};
  OIDNDevice oidn = oidnNewWebGPUDevice(&context);
  require(oidn, "external OIDN device creation failed");
  oidnCommitDevice(oidn);
  check(oidn);


  auto* engine = static_cast<oidn::WebGPUEngine*>(reinterpret_cast<oidn::WebGPUDevice*>(oidn)->getEngine());
  engine->profileConvMode = std::atoi(argv[5]);
  require(engine->profileConvMode >= 0 && engine->profileConvMode <= 2, "mode must be 0, 1 or 2");
  require(width > 0 && height > 0 && width <= 8192 && height <= 8192, "invalid dimensions");
  const size_t bytes = width * height * 12;
  std::vector<float> pixels(width * height * 3);
  for (size_t i=0; i<pixels.size(); ++i) pixels[i] = float((i*17+13)%251)/251.f;
  WGPUBuffer inputs[3]{};
  OIDNBuffer imported[3]{};
  for (int i=0; i<3; ++i) {
    inputs[i] = buffer(device, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    wgpuQueueWriteBuffer(queue, inputs[i], 0, pixels.data(), bytes);
    OIDNWebGPUBufferDesc desc{sizeof(desc), inputs[i], 0, bytes};
    imported[i] = oidnNewSharedBufferFromWebGPU(oidn, &desc);
  }
  auto output = buffer(device, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
  OIDNWebGPUBufferDesc desc{sizeof(desc), output, 0, bytes};
  OIDNBuffer out = oidnNewSharedBufferFromWebGPU(oidn, &desc);
  OIDNFilter filter = oidnNewFilter(oidn, "RT");
  oidnSetFilterImage(filter, "color", imported[0], OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  if (std::string(argv[1]).find("calb_cnrm") != std::string::npos) {
    oidnSetFilterImage(filter, "albedo", imported[1], OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterImage(filter, "normal", imported[2], OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  }
  oidnSetFilterImage(filter, "output", out, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  oidnSetFilterBool(filter, "srgb", true);
  oidnSetFilterBool(filter, "cleanAux", true);
  oidnSetFilterInt(filter, "maxMemoryMB", std::atoi(argv[6]));
  oidnSetSharedFilterData(filter, "weights", weights.data(), weights.size());
  require(oidnPrepareFilterWebGPU(filter), "prepare failed"); check(oidn);
  WGPUQuerySetDescriptor qdesc{}; qdesc.type = WGPUQueryType_Timestamp; qdesc.count = 4096;
  auto queries = wgpuDeviceCreateQuerySet(device, &qdesc);
  auto resolved = buffer(device, 4096*8, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
  auto readback = buffer(device, 4096*8, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
  const int repeats = std::atoi(argv[7]);
  require(repeats > 0 && repeats <= 1000, "repeats must be 1..1000");
  for (int run=0; run<=repeats; ++run) {
    auto encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    WGPUComputePassTimestampWrites stamps{}; stamps.querySet = queries;
    stamps.beginningOfPassWriteIndex = 0; stamps.endOfPassWriteIndex = WGPU_QUERY_SET_INDEX_UNDEFINED;
    WGPUComputePassDescriptor passDesc{}; passDesc.timestampWrites = &stamps;
    auto pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
    wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass);
    engine->profileLabels.clear(); engine->profileQueries = queries;
    auto start = std::chrono::steady_clock::now();
    require(oidnEncodeFilterWebGPU(filter, encoder), "encode failed"); check(oidn);
    require(validationErrors == 0, "Dawn validation failed during encode");
    double cpu = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    engine->profileQueries = nullptr;
    stamps.beginningOfPassWriteIndex = WGPU_QUERY_SET_INDEX_UNDEFINED; stamps.endOfPassWriteIndex = 1;
    pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
    wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass);
    uint32_t count = 2 + uint32_t(engine->profileLabels.size()) * 2;
    wgpuCommandEncoderResolveQuerySet(encoder, queries, 0, count, resolved, 0);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, resolved, 0, readback, 0, count*8);
    auto command = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(queue, 1, &command);
    wgpuCommandBufferRelease(command); wgpuCommandEncoderRelease(encoder);
    done = false; bool ok = false;
    WGPUBufferMapCallbackInfo2 cb{}; cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.userdata1 = &done; cb.userdata2 = &ok;
    cb.callback = [](WGPUMapAsyncStatus s,const char*,void* d,void* o) {
      *static_cast<bool*>(d)=true; *static_cast<bool*>(o)=s==WGPUMapAsyncStatus_Success;
    };
    wgpuBufferMapAsync2(readback,WGPUMapMode_Read,0,count*8,cb); wait(instance,done); require(ok,"map failed");
    auto* times = static_cast<const uint64_t*>(wgpuBufferGetConstMappedRange(readback,0,count*8));
    require(validationErrors == 0 && times && times[1] > times[0], "invalid GPU timestamps");
    double conv = 0;
    for(size_t i=0;i<engine->profileLabels.size();++i) {
      double ms = double(times[3+i*2]-times[2+i*2])*1e-6; conv += ms;
      if(run == repeats) std::printf("OP %s %.6f\n",engine->profileLabels[i].c_str(),ms);
    }
    std::printf("RUN %d gpu_ms=%.6f conv_ms=%.6f cpu_ms=%.6f dispatches=%zu\n",run,double(times[1]-times[0])*1e-6,conv,cpu,engine->profileLabels.size());
    std::fflush(stdout);
    wgpuBufferUnmap(readback);
  }
  require(validationErrors == 0, "Dawn validation failed");
  oidnReleaseFilter(filter); oidnReleaseBuffer(out);
  for(int i=0;i<3;++i) { oidnReleaseBuffer(imported[i]); wgpuBufferRelease(inputs[i]); }
  oidnReleaseDevice(oidn);
  wgpuBufferRelease(output); wgpuBufferRelease(resolved); wgpuBufferRelease(readback); wgpuQuerySetRelease(queries);
  wgpuQueueRelease(queue); wgpuDeviceRelease(device); wgpuAdapterRelease(adapter); wgpuInstanceRelease(instance);
  return 0;
}
catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
