#pragma once
#include "oidn.hpp"
#include <webgpu/webgpu.h>
#include <cstddef>

OIDN_NAMESPACE_BEGIN

  // Note: PascalCase to avoid collisions with Windows macros (CONST etc.)
  enum class WebGPUTensorType
  {
    Input,
    Const,
    Output
  };

  struct WebGPUTensor
  {
    WGPUBuffer buf;
    size_t     offset;
    uint32_t   n, c, h, w;
    WebGPUTensorType type;
    uint32_t   elemByteSize = 4; // 4 = f32, 2 = f16
  };

  // Parameters for the generalized conv2d kernel (see kConv2dWGSL)
  struct WebGPUConvParams
  {
    uint32_t padH       = 0;
    uint32_t padW       = 0;
    uint32_t activation = 0; // 0=none, 1=relu
    uint32_t biasMode   = 0; // 0=per-channel, 1=tensor, 2=tensor aliased with dst
    uint32_t postOp     = 0; // 0=none, 1=pool2x2, 2=upsample2x
  };

  class WebGPUEngine;

#if defined(_WIN32)
  #if defined(OpenImageDenoise_device_webgpu_EXPORTS)
    #define OIDN_WEBGPU_API __declspec(dllexport)
  #else
    #define OIDN_WEBGPU_API __declspec(dllimport)
  #endif
#else
  #define OIDN_WEBGPU_API __attribute__((visibility("default")))
#endif

  OIDN_WEBGPU_API WebGPUEngine* getEngine(const DeviceRef& device);

OIDN_NAMESPACE_END
