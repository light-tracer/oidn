// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "oidn.h"
#include <webgpu/webgpu.h>

#if defined(OIDN_STATIC_LIB)
  #define OIDN_WEBGPU_API
#elif defined(_WIN32)
  #if defined(OpenImageDenoise_device_webgpu_EXPORTS)
    #define OIDN_WEBGPU_API __declspec(dllexport)
  #else
    #define OIDN_WEBGPU_API __declspec(dllimport)
  #endif
#else
  #define OIDN_WEBGPU_API __attribute__((visibility("default")))
#endif

OIDN_API_NAMESPACE_BEGIN

typedef struct OIDNWebGPUContextDesc
{
  size_t structSize;
  WGPUDevice device;
  WGPUQueue queue; // NULL uses device's default queue
} OIDNWebGPUContextDesc;

typedef struct OIDNWebGPUBufferDesc
{
  size_t structSize;
  WGPUBuffer buffer;
  size_t byteOffset;
  size_t byteSize; // zero uses the rest of the buffer
} OIDNWebGPUBufferDesc;

typedef struct OIDNWebGPUTextureDesc
{
  size_t structSize;
  WGPUTexture texture;
  uint32_t mipLevel, arrayLayer;
  uint32_t originX, originY, width, height;
  // Output only: copy alpha from this texture (mip 0, layer 0, same region).
  // NULL writes alpha=1. The source requires TextureBinding usage.
  WGPUTexture alphaSource;
} OIDNWebGPUTextureDesc;

// References are retained, never destroyed. The runtime/header ABI must match
// the application's provider. Commit the returned OIDN device before using it.
OIDN_API_EXTERN_C OIDN_WEBGPU_API OIDNDevice oidnNewWebGPUDevice(const OIDNWebGPUContextDesc* desc);
OIDN_API_EXTERN_C OIDN_WEBGPU_API OIDNBuffer oidnNewSharedBufferFromWebGPU(OIDNDevice device, const OIDNWebGPUBufferDesc* desc);

// Imports a float/normalized 2D texture region as RGB. Inputs require
// TextureBinding; output requires StorageBinding and rgba8unorm, rgba16float
// or rgba32float. sRGB input formats undergo WebGPU's normal sRGB decoding.
// GPU conversion buffers are cached with the image; no CPU image staging.
// Use only with oidnEncodeFilterWebGPU. Handles are retained, never Destroyed.
OIDN_API_EXTERN_C OIDN_WEBGPU_API bool oidnSetFilterImageFromWebGPUTexture(
  OIDNFilter filter, const char* name, const OIDNWebGPUTextureDesc* desc);

// Preparation may enqueue weight uploads on the supplied queue; it never waits
// for unrelated application work. Call outside an encode. No Asyncify required.
OIDN_API_EXTERN_C OIDN_WEBGPU_API bool oidnPrepareFilterWebGPU(OIDNFilter filter);

// Records the entire filter, never finishes/submits/releases the borrowed encoder
// and never waits. Call outside any open pass. On failure discard that encoder.
// Encodes using a prepared filter must execute sequentially on the same queue.
OIDN_API_EXTERN_C OIDN_WEBGPU_API bool oidnEncodeFilterWebGPU(OIDNFilter filter, WGPUCommandEncoder encoder);

OIDN_API_NAMESPACE_END
