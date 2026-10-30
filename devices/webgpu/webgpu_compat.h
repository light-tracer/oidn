#pragma once
#include <webgpu/webgpu.h>
#include "common/platform.h"

OIDN_NAMESPACE_BEGIN
  const void* oidnWGPUBufferIdentity(WGPUBuffer buffer);
  const void* oidnWGPUTextureIdentity(WGPUTexture texture);
OIDN_NAMESPACE_END

// Light Tracer's Dawn uses SupportedLimits and callback API 2. Its browser
// bindings use Emscripten's older C API. Keep the provider ABI at this boundary.
#if defined(OIDN_WEBGPU_EXTERNAL_ONLY) && defined(__EMSCRIPTEN__)
  #define OIDN_WEBGPU_LEGACY_BROWSER
  using OIDNWGPUBufferUsage = WGPUBufferUsageFlags;
  using OIDNWGSLSource = WGPUShaderModuleWGSLDescriptor;
  using OIDNWGPUString = const char*;
  inline OIDNWGPUString oidnWGPUString(const char* s) { return s; }
  inline void oidnWGSLSource(OIDNWGSLSource& source, const char* code)
  { source.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor; source.code = code; }
#else
  using OIDNWGPUBufferUsage = WGPUBufferUsage;
  using OIDNWGSLSource = WGPUShaderSourceWGSL;
  using OIDNWGPUString = WGPUStringView;
  inline OIDNWGPUString oidnWGPUString(const char* s) { return {s, WGPU_STRLEN}; }
  inline void oidnWGSLSource(OIDNWGSLSource& source, const char* code)
  { source.chain.sType = WGPUSType_ShaderSourceWGSL; source.code = oidnWGPUString(code); }
#endif

#if defined(OIDN_WEBGPU_EXTERNAL_ONLY) && !defined(OIDN_WEBGPU_DAWN_MODERN)
  inline const char* oidnWGPUEntryString(const char* s) { return s; }
#else
  inline WGPUStringView oidnWGPUEntryString(const char* s) { return oidnWGPUString(s); }
#endif

#ifdef OIDN_WEBGPU_DAWN_MODERN
using WGPUComputePassTimestampWrites = WGPUPassTimestampWrites;
#endif
