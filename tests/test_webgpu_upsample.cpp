#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, Upsample2x)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t N=1,C=1,H=4,W=4;
  float src[N*C*H*W];
  for(size_t i=0;i<N*C*H*W;++i)
    src[i] = float(i+1);
  uint32_t OH = H*2, OW = W*2;
  float out[C*OH*OW];
  float ref[C*OH*OW];
  for(uint32_t y=0;y<OH;++y)
    for(uint32_t x=0;x<OW;++x)
      ref[y*OW+x] = src[(y/2)*W + (x/2)];

  auto A = eng->newTensor(src, WebGPUTensorType::INPUT, N,C,H,W);
  auto O = eng->newTensor(out, WebGPUTensorType::OUTPUT, C,1,OH,OW);

  eng->upsample2x(A,O);
  eng->sync();

  for(size_t i=0;i<C*OH*OW;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-6f);
}
