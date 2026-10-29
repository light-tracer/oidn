#include "OpenImageDenoise/oidn.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace oidn;

namespace
{
  void checkError(DeviceRef& device)
  {
    const char* message = nullptr;
    ASSERT_EQ(device.getError(message), Error::None) << (message ? message : "");
  }

  void compareFilter(int width, int height, bool hdr, int maxMemoryMB, bool inPlace)
  {
    if (!isWebGPUDeviceSupported() || !isCPUDeviceSupported())
      GTEST_SKIP();

    // RGBA-sized pixels with extra row padding and an image offset which is
    // float-aligned but not storage-binding-aligned. Padding must be preserved.
    const size_t pixelStride = 4;
    const size_t rowStride = size_t(width)*pixelStride + 8;
    const size_t offset = 3;
    const size_t count = offset + size_t(height)*rowStride;
    std::vector<float> color(count, -42.f), albedo(count, -42.f), normal(count, -42.f);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        for (int c = 0; c < 3; ++c)
        {
          const size_t i = offset + size_t(y)*rowStride + x*pixelStride + c;
          const float v = float((x*17 + y*29 + c*53) % 251) / 251.f;
          color[i] = v * (hdr ? 4.f : 1.f);
          albedo[i] = 0.2f + v*0.6f;
          normal[i] = v*2.f - 1.f;
        }

    DeviceRef cpu = newDevice(DeviceType::CPU);
    DeviceRef gpu = newWebGPUDevice();
    cpu.commit(); gpu.commit();
    checkError(cpu); checkError(gpu);

    auto run = [&](DeviceRef& device)
    {
      std::vector<float> out(count, -42.f);
      auto input = device.newBuffer(count*sizeof(float));
      auto alb = device.newBuffer(count*sizeof(float));
      auto nrm = device.newBuffer(count*sizeof(float));
      auto output = inPlace ? input : device.newBuffer(count*sizeof(float));
      auto filter = device.newFilter("RT");
      auto set = [&](const char* name, const BufferRef& buffer)
      {
        filter.setImage(name, buffer, Format::Float3, width, height,
                        offset*sizeof(float), pixelStride*sizeof(float), rowStride*sizeof(float));
      };
      set("color", input); set("albedo", alb); set("normal", nrm); set("output", output);
      filter.set("hdr", hdr);
      filter.set("srgb", !hdr);
      if (maxMemoryMB >= 0) filter.set("maxMemoryMB", maxMemoryMB);
      filter.commit();
      checkError(device);
      alb.write(0, count*sizeof(float), albedo.data());
      nrm.write(0, count*sizeof(float), normal.data());
      if (!inPlace) output.write(0, count*sizeof(float), out.data());

      // Exercise persistent filters, changing input/exposure, and uniform-arena
      // wrap-around without synchronizing between denoisings.
      for (int frame = 0; frame < 3; ++frame)
      {
        auto data = color;
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
              data[offset+size_t(y)*rowStride+x*pixelStride+c] *= 1.f+0.1f*frame;
        input.write(0, count*sizeof(float), data.data());
        filter.executeAsync();
        checkError(device);
      }
      device.sync();
      output.read(0, count*sizeof(float), out.data());
      checkError(device);
      return out;
    };

    auto expected = run(cpu);
    auto actual = run(gpu);
    double error = 0;
    for (size_t i = 0; i < count; ++i)
    {
      ASSERT_TRUE(std::isfinite(actual[i])) << i;
      if (expected[i] == -42.f)
        ASSERT_EQ(actual[i], expected[i]) << "padding at " << i;
      else
      {
        const double relative = std::abs(actual[i]-expected[i]) /
                                (std::abs(actual[i])+std::abs(expected[i])+0.01);
        error += relative;
        ASSERT_LE(relative, 0.05) << "pixel channel at " << i;
      }
    }
    EXPECT_LT(error / (size_t(width)*height*3), 0.003);
  }
}

TEST(WebGPU, FilterOddHDR) { compareFilter(333, 197, true, -1, false); }
TEST(WebGPU, FilterOddSRGBInPlace) { compareFilter(333, 197, false, -1, true); }
TEST(WebGPU, FilterTiledHDR) { compareFilter(1024, 1024, true, 200, false); }
TEST(WebGPU, FilterTiledSRGBInPlace) { compareFilter(1024, 1024, false, 200, true); }
