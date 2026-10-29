// SPDX-License-Identifier: Apache-2.0
#include <OpenImageDenoise/oidn_webgpu.h>
#include <emscripten.h>
#include <emscripten/html5_webgpu.h>
#include <cstdio>
#include <cstdlib>

static OIDNDevice device;
static OIDNFilter filter;
static OIDNBuffer in, out;
static bool check()
{
  const char* message = nullptr;
  if (oidnGetDeviceError(device, &message) == OIDN_ERROR_NONE) return true;
  std::fprintf(stderr, "OIDN: %s\n", message);
  return false;
}
EM_JS(void, exposeImports, (), {
  Module.importBuffer = value => WebGPU.mgrBuffer.create(value);
  Module.importEncoder = value => WebGPU.mgrCommandEncoder.create(value);
  Module.releaseBuffer = handle => WebGPU.mgrBuffer.release(handle);
  Module.releaseEncoder = handle => WebGPU.mgrCommandEncoder.release(handle);
  Module.importTexture = value => WebGPU.mgrTexture.create(value);
  Module.releaseTexture = handle => WebGPU.mgrTexture.release(handle);
});
extern "C" {
EMSCRIPTEN_KEEPALIVE bool oidnTestInit()
{
  exposeImports();
  WGPUDevice gpu = emscripten_webgpu_get_device();
  OIDNWebGPUContextDesc desc{sizeof(desc), gpu, nullptr};
  device = oidnNewWebGPUDevice(&desc);
  wgpuDeviceRelease(gpu);
  if (!device) return false;
  oidnCommitDevice(device);
  return check();
}
EMSCRIPTEN_KEEPALIVE bool oidnTestPrepare(WGPUBuffer color, WGPUBuffer output, void* weights, size_t bytes,
                                       int width, int height, bool hdr)
{
  if (filter) oidnReleaseFilter(filter);
  if (in) oidnReleaseBuffer(in);
  if (out) oidnReleaseBuffer(out);
  OIDNWebGPUBufferDesc desc{sizeof(desc), color, 0, 0};
  in = oidnNewSharedBufferFromWebGPU(device, &desc);
  desc.buffer = output;
  out = oidnNewSharedBufferFromWebGPU(device, &desc);
  filter = oidnNewFilter(device, "RT");
  oidnSetFilterImage(filter, "color", in, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  oidnSetFilterImage(filter, "output", out, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  oidnSetFilterBool(filter, "hdr", hdr);
  oidnSetFilterBool(filter, "srgb", !hdr);
  if (width >= 1024) oidnSetFilterInt(filter, "maxMemoryMB", 64);
  oidnSetSharedFilterData(filter, "weights", weights, bytes);
  return oidnPrepareFilterWebGPU(filter) && check();
}
EMSCRIPTEN_KEEPALIVE bool oidnTestEncode(WGPUCommandEncoder encoder)
{
  return oidnEncodeFilterWebGPU(filter, encoder) && check();
}
EMSCRIPTEN_KEEPALIVE bool oidnTestPrepareTexture(WGPUTexture color,WGPUTexture output,void* weights,size_t bytes,
                                               int width,int height,bool hdr)
{
  if (filter) oidnReleaseFilter(filter);
  filter=oidnNewFilter(device,"RT");
  OIDNWebGPUTextureDesc desc{};
  desc.structSize=sizeof(desc); desc.texture=color; desc.width=width; desc.height=height;
  if(!oidnSetFilterImageFromWebGPUTexture(filter,"color",&desc)) return check() && false;
  desc.texture=output; desc.alphaSource=color;
  if(!oidnSetFilterImageFromWebGPUTexture(filter,"output",&desc)) return check() && false;
  oidnSetFilterBool(filter,"hdr",hdr);
  oidnSetFilterBool(filter,"srgb",!hdr);
  if(width>=1024) oidnSetFilterInt(filter,"maxMemoryMB",64);
  oidnSetSharedFilterData(filter,"weights",weights,bytes);
  return oidnPrepareFilterWebGPU(filter) && check();
}
EMSCRIPTEN_KEEPALIVE void* oidnTestAlloc(size_t bytes) { return std::malloc(bytes); }
EMSCRIPTEN_KEEPALIVE void oidnTestFree(void* ptr) { std::free(ptr); }
EMSCRIPTEN_KEEPALIVE void oidnTestShutdown()
{
  if(filter) oidnReleaseFilter(filter);
  if(in) oidnReleaseBuffer(in);
  if(out) oidnReleaseBuffer(out);
  check();
  if(device) oidnReleaseDevice(device);
  filter=nullptr; in=out=nullptr; device=nullptr;
}
}
