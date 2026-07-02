#include "OpenImageDenoise/oidn.hpp"
#include "OpenImageDenoise/webgpu.h"
#include "../devices/webgpu/webgpu_engine.h"
#include "../core/concat_conv_hwc.h"
#include "../core/tensor.h"
#include "../common/platform.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace oidn;

// Validates ConcatConvHWC (used by the UNet decoder for skip connections) on
// the WebGPU backend against a plain host reference: channel-concat of two
// HWC tensors followed by a same-padding conv + bias + ReLU.
TEST(WebGPU, ConcatConv)
{
  if (!isWebGPUDeviceSupported())
    GTEST_SKIP();

  auto dev = newWebGPUDevice();
  dev.commit();
  WebGPUEngine* eng = getEngine(dev);

  constexpr int C1=2, C2=3, H=8, W=8;
  constexpr int OC=2, KH=3, KW=3;
  constexpr int IC = C1 + C2;

  std::vector<float> src1(H*W*C1), src2(H*W*C2), weight(OC*IC*KH*KW), bias(OC);
  for (size_t i=0;i<src1.size();++i)   src1[i]   = std::sin(float(i)*0.37f);
  for (size_t i=0;i<src2.size();++i)   src2[i]   = std::cos(float(i)*0.23f);
  for (size_t i=0;i<weight.size();++i) weight[i] = std::sin(float(i)*0.11f) * 0.5f;
  for (size_t i=0;i<bias.size();++i)   bias[i]   = 0.1f * float(i+1);

  // Host reference (HWC layout)
  std::vector<float> ref(H*W*OC);
  auto srcAt = [&](int y, int x, int c) -> float
  {
    if (y < 0 || y >= H || x < 0 || x >= W) return 0.f;
    return (c < C1) ? src1[(y*W + x)*C1 + c] : src2[(y*W + x)*C2 + (c - C1)];
  };
  for (int y=0;y<H;++y)
    for (int x=0;x<W;++x)
      for (int o=0;o<OC;++o)
      {
        float acc = bias[o];
        for (int c=0;c<IC;++c)
          for (int ky=0;ky<KH;++ky)
            for (int kx=0;kx<KW;++kx)
              acc += srcAt(y+ky-1, x+kx-1, c) * weight[((o*IC + c)*KH + ky)*KW + kx];
        ref[(y*W + x)*OC + o] = std::max(acc, 0.f);
      }

  // Device tensors
  TensorDesc src1Desc({C1,H,W}, TensorLayout::hwc, DataType::Float32);
  TensorDesc src2Desc({C2,H,W}, TensorLayout::hwc, DataType::Float32);
  TensorDesc weightDesc({OC,IC,KH,KW}, TensorLayout::oihw, DataType::Float32);
  TensorDesc biasDesc({OC}, TensorLayout::x, DataType::Float32);

  ConcatConvDesc desc{src1Desc, src2Desc, weightDesc, biasDesc, Activation::ReLU, false};
  ConcatConvHWC concatConv(eng, desc);

  // Split weights: conv1 gets input channels [0,C1), conv2 gets [C1,IC)
  std::vector<float> w1(OC*C1*KH*KW), w2(OC*C2*KH*KW);
  for (int o=0;o<OC;++o)
    for (int c=0;c<IC;++c)
      for (int k=0;k<KH*KW;++k)
      {
        float v = weight[(o*IC + c)*KH*KW + k];
        if (c < C1) w1[(o*C1 + c)*KH*KW + k] = v;
        else        w2[(o*C2 + (c-C1))*KH*KW + k] = v;
      }

  auto makeDevTensor = [&](const TensorDesc& td, const void* data, size_t byteSize)
  {
    auto buf = dev.newBuffer(byteSize);
    if (data)
      buf.write(0, byteSize, data);
    return eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(buf.getHandle())), td);
  };

  auto src1T = makeDevTensor(src1Desc, src1.data(), src1.size()*sizeof(float));
  auto src2T = makeDevTensor(src2Desc, src2.data(), src2.size()*sizeof(float));
  auto w1T   = makeDevTensor(concatConv.getWeight1Desc(), w1.data(), w1.size()*sizeof(float));
  auto w2T   = makeDevTensor(concatConv.getWeight2Desc(), w2.data(), w2.size()*sizeof(float));
  auto biasT = makeDevTensor(biasDesc, bias.data(), bias.size()*sizeof(float));

  auto dstDesc = concatConv.getDstDesc();
  ASSERT_EQ(dstDesc.getH(), H);
  ASSERT_EQ(dstDesc.getW(), W);
  ASSERT_EQ(dstDesc.getC(), OC);
  auto dstBuf = dev.newBuffer(dstDesc.getByteSize());
  auto dstT = eng->Engine::newTensor(Ref<Buffer>(reinterpret_cast<Buffer*>(dstBuf.getHandle())), dstDesc);

  concatConv.setSrc(src1T, src2T);
  concatConv.setWeight(w1T, w2T);
  concatConv.setBias(biasT);
  concatConv.setDst(dstT);
  concatConv.finalize();
  concatConv.submit(nullptr);
  dev.sync();

  std::vector<float> out(ref.size());
  dstBuf.read(0, out.size()*sizeof(float), out.data());

  for (size_t i=0;i<ref.size();++i)
    ASSERT_NEAR(out[i], ref[i], 1e-4f * std::max(1.f, std::abs(ref[i]))) << "at index " << i;
}
