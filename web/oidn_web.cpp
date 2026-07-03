// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Minimal C API for the Open Image Denoise WebGPU backend, intended to be
// called from JavaScript via Emscripten ccall/cwrap. Keeps a single persistent
// device and caches the filter + buffers across calls with identical settings,
// so repeated denoises skip the expensive weights parsing/reordering/upload.
//
// The web build ships no embedded weights (OIDN_FILTER_RT=OFF keeps the wasm
// small); a weights blob (.tza) must be provided via oidnWebSetWeights before
// the first denoise.

#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "devices/webgpu/webgpu_buffer.h"
#include "devices/webgpu/webgpu_device.h"
#include <emscripten.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

using namespace oidn;

// Publishes the device and output buffer JS objects to the page (via the
// emdawnwebgpu object table) so it can display the result directly from the
// GPU — e.g. by rendering the buffer to a WebGPU canvas — without any
// readback to the CPU.
EM_JS(void, oidnWebPublishOutput,
      (WGPUDevice devicePtr, WGPUBuffer bufferPtr, uint32_t byteOffset,
       uint32_t byteSize, int width, int height),
{
  Module.oidnOutput = {
    device: WebGPU.getJsObject(devicePtr),
    buffer: WebGPU.getJsObject(bufferPtr),
    offset: byteOffset,
    byteSize: byteSize,
    width: width,
    height: height,
  };
});

namespace {
  DeviceRef g_device;

  std::vector<char> g_weights;
  int g_weightsGeneration = 0;

  // Cached filter state
  FilterRef g_filter;
  BufferRef g_colorBuf, g_albedoBuf, g_normalBuf, g_outputBuf;
  int g_width = 0, g_height = 0, g_hdr = -1;
  bool g_hasAlbedo = false, g_hasNormal = false;
  int g_filterWeightsGeneration = -1;

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

  void releaseFilter()
  {
    g_filter    = FilterRef();
    g_colorBuf  = BufferRef();
    g_albedoBuf = BufferRef();
    g_normalBuf = BufferRef();
    g_outputBuf = BufferRef();
    g_width = g_height = 0;
    g_hdr = -1;
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

// Set the weights blob (.tza contents) used by subsequent denoises. The data
// is copied, so the caller may free it right away. Returns 1 on success.
EMSCRIPTEN_KEEPALIVE
int oidnWebSetWeights(const void* data, int byteSize)
{
  if (!data || byteSize <= 0) { printf("OIDN: invalid weights\n"); return 0; }
  g_weights.assign(static_cast<const char*>(data),
                   static_cast<const char*>(data) + byteSize);
  g_weightsGeneration++;
  return 1;
}

// Denoise a width x height image with the RT filter.
//
// color/albedo/normal point at interleaved RGB float32 data in the wasm heap
// (3 channels, row-major, top row first). albedo and normal may be null.
// output receives RGB float32; it may be null, in which case the result stays
// on the GPU and is published to JS as Module.oidnOutput (device + buffer),
// letting the page display it directly (no readback). hdr != 0 marks the
// color image as HDR linear. Returns 1 on success, 0 on failure.
EMSCRIPTEN_KEEPALIVE
int oidnWebDenoise(int width, int height,
                   const float* color, const float* albedo, const float* normal,
                   float* output, int hdr)
{
  if (!g_device) { printf("OIDN: device not initialized\n"); return 0; }
  if (g_weights.empty()) { printf("OIDN: weights not set (call oidnWebSetWeights)\n"); return 0; }

  try
  {
    const size_t bytes = size_t(width) * size_t(height) * 3 * sizeof(float);

    const bool reuse = g_filter &&
                       width  == g_width &&
                       height == g_height &&
                       hdr    == g_hdr &&
                       (albedo != nullptr) == g_hasAlbedo &&
                       (normal != nullptr) == g_hasNormal &&
                       g_filterWeightsGeneration == g_weightsGeneration;

    if (!reuse)
    {
      releaseFilter();

      g_filter = g_device.newFilter("RT");

      g_colorBuf = g_device.newBuffer(bytes);
      g_filter.setImage("color", g_colorBuf, Format::Float3, width, height);

      if (albedo)
      {
        g_albedoBuf = g_device.newBuffer(bytes);
        g_filter.setImage("albedo", g_albedoBuf, Format::Float3, width, height);
      }
      if (normal)
      {
        g_normalBuf = g_device.newBuffer(bytes);
        g_filter.setImage("normal", g_normalBuf, Format::Float3, width, height);
      }

      g_outputBuf = g_device.newBuffer(bytes);
      g_filter.setImage("output", g_outputBuf, Format::Float3, width, height);

      if (hdr)
        g_filter.set("hdr", true);

      g_filter.setData("weights", g_weights.data(), g_weights.size());

      g_filter.commit();
      if (!checkError("commit")) { releaseFilter(); return 0; }

      g_width = width; g_height = height; g_hdr = hdr;
      g_hasAlbedo = albedo != nullptr;
      g_hasNormal = normal != nullptr;
      g_filterWeightsGeneration = g_weightsGeneration;
    }

    const double t0 = emscripten_get_now();

    g_colorBuf.write(0, bytes, color);
    if (albedo) g_albedoBuf.write(0, bytes, albedo);
    if (normal) g_normalBuf.write(0, bytes, normal);

    const double t1 = emscripten_get_now();

    g_filter.executeAsync();
    if (!checkError("execute")) return 0;

    const double t2 = emscripten_get_now();

    if (output)
    {
      // CPU readback path: wait for the GPU, then map + copy
      g_device.sync();
      const double t3 = emscripten_get_now();
      g_outputBuf.read(0, bytes, output);
      const double t4 = emscripten_get_now();
      printf("oidn: upload %.1f ms, submit %.1f ms, gpu %.1f ms, readback %.1f ms\n",
             t1 - t0, t2 - t1, t3 - t2, t4 - t3);
    }
    else
    {
      // GPU-only path: flush the recorded graph to the queue and publish the
      // output buffer to JS; the page renders it directly (queue-ordered), so
      // no wait or readback is needed here
      auto* engine = getEngine(g_device);
      auto* dev = static_cast<WebGPUDevice*>(engine->getDevice());
      auto* buf = dynamic_cast<WebGPUBuffer*>(
        reinterpret_cast<Buffer*>(g_outputBuf.getHandle()));
      if (!dev || !buf) { printf("OIDN: cannot publish output\n"); return 0; }

      engine->flush();
      oidnWebPublishOutput(dev->device, buf->getWGPUBuffer(),
                           uint32_t(buf->getWGPUBufferOffset()), uint32_t(bytes),
                           width, height);
      const double t3 = emscripten_get_now();
      printf("oidn: upload %.1f ms, submit %.1f ms, flush+publish %.1f ms (display is async)\n",
             t1 - t0, t2 - t1, t3 - t2);
    }
    return 1;
  }
  catch (const std::exception& e)
  {
    printf("OIDN denoise exception: %s\n", e.what());
    releaseFilter();
    return 0;
  }
}

// Release the filter and device.
EMSCRIPTEN_KEEPALIVE
void oidnWebShutdown()
{
  releaseFilter();
  g_device = DeviceRef();
}

} // extern "C"
