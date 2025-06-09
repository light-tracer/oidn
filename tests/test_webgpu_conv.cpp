#include "webgpu.h"
#include <gtest/gtest.h>

using namespace oidn;

TEST(WebGPU, Conv2d)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  const uint32_t N=1,C=1,H=8,W=8;
  const uint32_t OC=1,IC=1,KH=3,KW=3;
  float src[N*C*H*W];
  float weight[OC*IC*KH*KW];
  float bias[OC];
  for(size_t i=0;i<N*C*H*W;++i) src[i]= (float)i*0.01f + 1.0f;
  for(size_t i=0;i<OC*IC*KH*KW;++i) weight[i]= (float)(i+1)*0.1f;
  bias[0]=0.5f;
  uint32_t OH=H-KH+1; uint32_t OW=W-KW+1;
  float out[OC*OH*OW];
  float ref[OC*OH*OW];
  for(uint32_t y=0;y<OH;++y)
    for(uint32_t x=0;x<OW;++x)
    {
      float acc=0.f;
      for(uint32_t ky=0;ky<KH;++ky)
        for(uint32_t kx=0;kx<KW;++kx)
          acc+=src[(y+ky)*W+(x+kx)]*weight[ky*KW+kx];
      acc+=bias[0];
      if(acc<0.f) acc=0.f;
      ref[y*OW+x]=acc;
    }

  auto A = eng->newTensor(src, WebGPUTensorType::INPUT, N,C,H,W);
  auto Wt= eng->newTensor(weight, WebGPUTensorType::CONST, OC,IC,KH,KW);
  auto B = eng->newTensor(bias, WebGPUTensorType::CONST, OC,1,1,1);
  auto O = eng->newTensor(out, WebGPUTensorType::OUTPUT, OC,1,OH,OW);

  eng->conv2d_eltwise(A,Wt,B,O);
  eng->sync();

  for(size_t i=0;i<OC*OH*OW;++i)
    ASSERT_NEAR(out[i], ref[i], 1e-4f);
}

