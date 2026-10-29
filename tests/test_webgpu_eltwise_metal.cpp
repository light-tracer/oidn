#include "OpenImageDenoise/oidn.hpp"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, EltwiseMetal)
{
#if !defined(OIDN_DEVICE_METAL)
  GTEST_SKIP() << "Metal backend not built";
#else
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();
  if (!isMetalDeviceSupported(nullptr))
    GTEST_SKIP();
  // TODO: compare WebGPU eltwise kernels against Metal implementation
  FAIL() << "Metal comparison not implemented";
#endif
}
