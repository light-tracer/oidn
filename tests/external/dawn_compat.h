#pragma once
#include <webgpu/webgpu.h>
#include <string>
#ifdef OIDN_WEBGPU_DAWN_MODERN
using TestMessage = WGPUStringView;
inline std::string testText(WGPUStringView s) { return s.data ? std::string(s.data, s.length == WGPU_STRLEN ? std::char_traits<char>::length(s.data) : s.length) : ""; }
#define WGPURequestAdapterCallbackInfo2 WGPURequestAdapterCallbackInfo
#define WGPURequestDeviceCallbackInfo2 WGPURequestDeviceCallbackInfo
#define WGPUBufferMapCallbackInfo2 WGPUBufferMapCallbackInfo
#define wgpuInstanceRequestAdapter2 wgpuInstanceRequestAdapter
#define wgpuAdapterRequestDevice2 wgpuAdapterRequestDevice
#define wgpuBufferMapAsync2 wgpuBufferMapAsync
#define uncapturedErrorCallbackInfo2 uncapturedErrorCallbackInfo
#define WGPUComputePassTimestampWrites WGPUPassTimestampWrites
#define WGPUImageCopyTexture WGPUTexelCopyTextureInfo
#define WGPUTextureDataLayout WGPUTexelCopyBufferLayout
#define WGPUImageCopyBuffer WGPUTexelCopyBufferInfo
#else
using TestMessage = const char*;
inline std::string testText(const char* s) { return s ? s : ""; }
#endif
