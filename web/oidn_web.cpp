// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Minimal C API for the Open Image Denoise WebGPU backend, intended to be
// called from JavaScript via Emscripten ccall/cwrap. Keeps a single persistent
// device; each denoise call builds an RT filter over device buffers.

#include "OpenImageDenoise/oidn.hpp"
#include <emscripten.h>
#include <cstdio>
#include <exception>

using namespace oidn;

namespace {
  DeviceRef g_device;

  bool checkError(const char* stage)
  {
    const char* msg = nullptr;
    if (g_device.getError(msg) != Error::None)
    {
      printf("OIDN %s error: %s\n", stage, msg ? msg : "(unknown)");
      return false;
    }
    return true;
  }
}

extern "C" {

// Create and commit the WebGPU device. Returns 1 on success, 0 on failure.
// Must be called (and awaited) before oidnWebDenoise. Blocks internally via
// Asyncify while the browser resolves the adapter/device requests.
EMSCRIPTEN_KEEPALIVE
int oidnWebInit()
{
  try
  {
    g_device = newDevice(DeviceType::WGPU);
    if (!g_device) { printf("OIDN: WebGPU device not available\n"); return 0; }
    g_device.commit();
    return checkError("init") ? 1 : 0;
  }
  catch (const std::exception& e)
  {
    printf("OIDN init exception: %s\n", e.what());
    return 0;
  }
}

// Denoise a width x height image with the RT filter.
//
// color/albedo/normal are byte offsets into the wasm heap pointing at
// interleaved RGB float32 data (3 channels, row-major, top row first).
// albedo and normal may be 0 (absent). output receives RGB float32.
// hdr != 0 marks the color image as HDR linear (enables autoexposure).
// Returns 1 on success, 0 on failure.
EMSCRIPTEN_KEEPALIVE
int oidnWebDenoise(int width, int height,
                   const float* color, const float* albedo, const float* normal,
                   float* output, int hdr)
{
  if (!g_device) { printf("OIDN: device not initialized\n"); return 0; }

  try
  {
    const size_t bytes = size_t(width) * size_t(height) * 3 * sizeof(float);

    FilterRef filter = g_device.newFilter("RT");

    BufferRef colorBuf = g_device.newBuffer(bytes);
    colorBuf.write(0, bytes, color);
    filter.setImage("color", colorBuf, Format::Float3, width, height);

    BufferRef albedoBuf, normalBuf;
    if (albedo)
    {
      albedoBuf = g_device.newBuffer(bytes);
      albedoBuf.write(0, bytes, albedo);
      filter.setImage("albedo", albedoBuf, Format::Float3, width, height);
    }
    if (normal)
    {
      normalBuf = g_device.newBuffer(bytes);
      normalBuf.write(0, bytes, normal);
      filter.setImage("normal", normalBuf, Format::Float3, width, height);
    }

    BufferRef outputBuf = g_device.newBuffer(bytes);
    filter.setImage("output", outputBuf, Format::Float3, width, height);

    if (hdr)
      filter.set("hdr", true);

    filter.commit();
    if (!checkError("commit")) return 0;

    filter.execute();
    if (!checkError("execute")) return 0;

    outputBuf.read(0, bytes, output);
    return 1;
  }
  catch (const std::exception& e)
  {
    printf("OIDN denoise exception: %s\n", e.what());
    return 0;
  }
}

// Release the device.
EMSCRIPTEN_KEEPALIVE
void oidnWebShutdown()
{
  g_device = DeviceRef();
}

} // extern "C"
