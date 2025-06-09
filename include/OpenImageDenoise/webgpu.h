#pragma once
#include "oidn.hpp"
#include <webgpu/webgpu.h>
#include <cstddef>

OIDN_NAMESPACE_BEGIN

  enum class WebGPUTensorType
  {
    INPUT,
    CONST,
    OUTPUT
  };

  struct WebGPUTensor
  {
    WGPUBuffer buf;
    size_t     offset;
    uint32_t   n, c, h, w;
    WebGPUTensorType type;
  };

  class WebGPUEngine;

  WebGPUEngine* getEngine(DeviceRef device);

OIDN_NAMESPACE_END
