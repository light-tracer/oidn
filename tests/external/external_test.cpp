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
void testExternalUniforms(OIDNDevice, WGPUDevice, WGPUQueue, WGPUInstance);
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

int main(int argc, char** argv) try
{
  if (argc != 8 && argc != 9)
  {
    std::fprintf(stderr, "Usage: oidn_external_test weights.tza input.bin reference.bin width height srgb|hdr f16|f32\n");
    return 2;
  }
  auto weights = load(argv[1]);
  const bool textures = argc == 9 && std::strcmp(argv[8], "texture") == 0;
  const auto input = load(argv[2]);
  const auto reference = load(argv[3]);
  const size_t width = std::atoi(argv[4]), height = std::atoi(argv[5]);
  const bool hdr = std::strcmp(argv[6], "hdr") == 0;
  require(input.size() == width * height * 12 && reference.size() == input.size(), "fixture size mismatch");

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
  wgpuInstanceRequestAdapter2(instance, &adapterOptions, adapterCB);
  wait(instance, done);
  require(adapter, "no Dawn adapter");

  WGPUDevice device = nullptr;
  done = false;
  WGPUDeviceDescriptor deviceDesc{};
  WGPUFeatureName feature = WGPUFeatureName_ShaderF16;
  if (std::strcmp(argv[7], "f16") == 0)
  {
    require(wgpuAdapterHasFeature(adapter, feature), "adapter has no shader-f16");
    deviceDesc.requiredFeatureCount = 1;
    deviceDesc.requiredFeatures = &feature;
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

  if (!textures && !hdr && width == 333) testExternalUniforms(oidn, device, queue, instance);

  // Imported views with an unaligned storage-binding base and padded float4 rows.
  const size_t pixelStride = 16;
  const size_t rowStride = textures ? ((width + 4) * 16 + 255) / 256 * 256 : width * 16 + 32;
  const size_t offset = textures ? rowStride + 32 : 20;
  const size_t bytes = textures ? rowStride * (height + 2) : offset + rowStride * height + 256;
  std::vector<float> padded(bytes / 4, -123.f);
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x)
      std::memcpy(reinterpret_cast<char*>(padded.data()) + offset + y * rowStride + x * pixelStride,
                  input.data() + (y * width + x) * 12, 12);
  WGPUBuffer color = buffer(device, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
  WGPUBuffer output = buffer(device, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
  wgpuQueueWriteBuffer(queue, color, 0, padded.data(), bytes);
  wgpuQueueWriteBuffer(queue, output, 0, padded.data(), bytes);
  OIDNWebGPUBufferDesc desc{sizeof(desc), color, offset, bytes - offset};
  OIDNBuffer in = oidnNewSharedBufferFromWebGPU(oidn, &desc);
  desc.buffer = output;
  OIDNBuffer out = oidnNewSharedBufferFromWebGPU(oidn, &desc);
  check(oidn);
  require(in && out, "import failed");
  OIDNWebGPUBufferDesc invalid=desc;
  invalid.byteOffset=bytes+4;
  require(!oidnNewSharedBufferFromWebGPU(oidn,&invalid),"out-of-range import was accepted");
  require(oidnGetDeviceError(oidn,nullptr)==OIDN_ERROR_INVALID_ARGUMENT,"wrong import error");
  WGPUBuffer noStorage=buffer(device,256,WGPUBufferUsage_CopySrc);
  invalid={sizeof(invalid),noStorage,0,0};
  require(!oidnNewSharedBufferFromWebGPU(oidn,&invalid),"missing storage usage was accepted");
  require(oidnGetDeviceError(oidn,nullptr)==OIDN_ERROR_INVALID_ARGUMENT,"wrong storage error");
  wgpuBufferRelease(noStorage);
  OIDNFilter filter = oidnNewFilter(oidn, "RT");
  oidnSetFilterImage(filter, "color", in, OIDN_FORMAT_FLOAT3, width, height, 0, pixelStride, rowStride);
  oidnSetFilterImage(filter, "output", out, OIDN_FORMAT_FLOAT3, width, height, 0, pixelStride, rowStride);
  WGPUTexture colorTexture = nullptr, outputTexture = nullptr;
  WGPUImageCopyTexture textureCopy{};
  WGPUExtent3D extent{uint32_t(width + 4),uint32_t(height + 2),1};
  if (textures)
  {
    WGPUTextureDescriptor td{};
    td.dimension = WGPUTextureDimension_2D;
    td.size = extent; td.mipLevelCount = 1; td.sampleCount = 1;
    td.format = WGPUTextureFormat_RGBA32Float;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    colorTexture = wgpuDeviceCreateTexture(device, &td);
    td.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc;
    outputTexture = wgpuDeviceCreateTexture(device, &td);
    textureCopy.texture = colorTexture; textureCopy.aspect = WGPUTextureAspect_All;
    WGPUTextureDataLayout layout{};
    layout.bytesPerRow = uint32_t(rowStride); layout.rowsPerImage = extent.height;
    wgpuQueueWriteTexture(queue,&textureCopy,padded.data(),bytes,&layout,&extent);
    textureCopy.texture = outputTexture;
    wgpuQueueWriteTexture(queue,&textureCopy,padded.data(),bytes,&layout,&extent);
    OIDNWebGPUTextureDesc image{};
    image.structSize = sizeof(image); image.texture = colorTexture;
    image.originX = 2; image.originY = 1; image.width = uint32_t(width); image.height = uint32_t(height);
    require(oidnSetFilterImageFromWebGPUTexture(filter,"color",&image),"texture input failed");
    image.texture = outputTexture; image.alphaSource = colorTexture;
    require(oidnSetFilterImageFromWebGPUTexture(filter,"output",&image),"texture output failed");
  }
  oidnSetFilterBool(filter, "hdr", hdr);
  oidnSetFilterBool(filter, "srgb", !hdr);
  if (width >= 1024) oidnSetFilterInt(filter, "maxMemoryMB", 512);
  oidnSetSharedFilterData(filter, "weights", weights.data(), weights.size());
  require(oidnPrepareFilterWebGPU(filter), "prepare failed");
  check(oidn);
  require(!oidnEncodeFilterWebGPU(filter,nullptr),"null encoder was accepted");
  require(oidnGetDeviceError(oidn,nullptr)==OIDN_ERROR_INVALID_OPERATION,"wrong encoder error");

  WGPUCommandBuffer commands[2]{};
  WGPUBuffer readback[2]{};
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 2; ++i)
  {
    if (i == 1 && width >= 1024)
    {
      // Change graph/tile uniforms while the first encoder is still unsubmitted.
      // Reusing uniform slots would corrupt that earlier execution.
      oidnSetFilterInt(filter, "maxMemoryMB", 64);
      require(oidnPrepareFilterWebGPU(filter), "reprepare before submit failed");
      check(oidn);
    }
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    require(oidnEncodeFilterWebGPU(filter, encoder), "encode failed");
    check(oidn);
    readback[i] = buffer(device, bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    if (textures)
    {
      WGPUImageCopyBuffer destination{};
      destination.buffer = readback[i]; destination.layout.bytesPerRow = uint32_t(rowStride);
      destination.layout.rowsPerImage = extent.height;
      wgpuCommandEncoderCopyTextureToBuffer(encoder,&textureCopy,&destination,&extent);
    }
    else wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback[i], 0, bytes);
    commands[i] = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuCommandEncoderRelease(encoder);
  }
  const double encodeMS = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  // Command buffers must retain all objects, even after the caller releases OIDN.
  oidnReleaseFilter(filter);
  oidnReleaseBuffer(in);
  oidnReleaseBuffer(out);
  check(oidn);
  oidnReleaseDevice(oidn);
  wgpuBufferRelease(color);
  wgpuBufferRelease(output);
  if (colorTexture) wgpuTextureRelease(colorTexture);
  if (outputTexture) wgpuTextureRelease(outputTexture);
  wgpuQueueSubmit(queue, 2, commands);
  for (int i = 0; i < 2; ++i)
  {
    wgpuCommandBufferRelease(commands[i]);
    bool mapped = false;
    done = false;
    WGPUBufferMapCallbackInfo2 mapCB{};
    mapCB.mode = WGPUCallbackMode_AllowProcessEvents;
    mapCB.userdata1 = &done;
    mapCB.userdata2 = &mapped;
    mapCB.callback = [](WGPUMapAsyncStatus status, const char* message, void* done, void* mapped)
    {
      *static_cast<bool*>(mapped) = status == WGPUMapAsyncStatus_Success;
      *static_cast<bool*>(done) = true;
      if (status != WGPUMapAsyncStatus_Success) std::fprintf(stderr, "Map: %s\n", message);
    };
    wgpuBufferMapAsync2(readback[i], WGPUMapMode_Read, 0, bytes, mapCB);
    wait(instance, done);
    require(mapped, "map failed");
    const auto* result = static_cast<const char*>(wgpuBufferGetConstMappedRange(readback[i], 0, bytes));
    double error = 0;
    size_t outliers = 0;
    for (size_t j = 0; j < padded.size(); ++j)
    {
      const size_t b = j * 4;
      if (textures && b % rowStride >= (width + 4) * 16) continue;
      const size_t row = b >= offset ? (b - offset) / rowStride : height;
      const size_t column = b >= offset ? (b - offset) % rowStride : rowStride;
      const float value = reinterpret_cast<const float*>(result)[j];
      if (row < height && column < width * pixelStride && column % pixelStride < 12)
      {
        const float ref = reinterpret_cast<const float*>(reference.data())[row * width * 3 + (column / pixelStride) * 3 + column % pixelStride / 4];
        const float absolute = std::abs(value - ref), relative = absolute / (std::abs(value) + std::abs(ref) + .01f);
        require(std::isfinite(value), "non-finite output");
        error += relative;
        outliers += absolute > .02f && relative > .05f;
      }
      else require(value == padded[j], "padding or alpha overwritten");
    }
    error /= width * height * 3;
    std::printf("%zux%zu %s encode %d: relative mean %.8f, outliers %zu; CPU encode %.2f ms\n", width, height, argv[7], i, error, outliers, encodeMS / 2);
    require(error <= .003 && outliers == 0, "CPU parity failed");
    wgpuBufferUnmap(readback[i]);
    wgpuBufferRelease(readback[i]);
  }
  require(validationErrors == 0, "Dawn validation failed");
  wgpuQueueRelease(queue);
  wgpuDeviceRelease(device);
  wgpuAdapterRelease(adapter);
  wgpuInstanceRelease(instance);
  return 0;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
