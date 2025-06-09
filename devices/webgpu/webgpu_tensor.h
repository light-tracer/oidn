#pragma once
#include <webgpu/webgpu.h>
#include <cstdint>

OIDN_NAMESPACE_BEGIN

  struct WebGPUTensor
  {
    WGPUBuffer buf;
    size_t     offset;
    uint32_t   n, c, h, w;
  };

  enum class WebGPUTensorType
  {
    INPUT,
    CONST,
    OUTPUT
  };

OIDN_NAMESPACE_END
