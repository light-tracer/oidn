#include "webgpu_autoexposure.h"
#include "core/color.h"
#include <vector>
#include <cmath>

OIDN_NAMESPACE_BEGIN

  WebGPUAutoexposure::WebGPUAutoexposure(WebGPUEngine* engine, const ImageDesc& srcDesc)
    : Autoexposure(srcDesc), engine(engine) {}

  static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

  void WebGPUAutoexposure::submitKernels(const Ref<CancellationToken>&)
  {
    if (!src || !dst)
      throw std::logic_error("autoexposure source/destination not set");
    if (src->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");

    const float* img = static_cast<const float*>(src->getPtr());
    const int H = srcDesc.getH();
    const int W = srcDesc.getW();
    const int numBinsH = ceil_div_int(H, maxBinSize);
    const int numBinsW = ceil_div_int(W, maxBinSize);

    double logSum = 0.0;
    int    count  = 0;

    for (int i=0;i<numBinsH;++i)
      for (int j=0;j<numBinsW;++j)
      {
        int beginH = i    * H / numBinsH;
        int endH   = (i+1)* H / numBinsH;
        int beginW = j    * W / numBinsW;
        int endW   = (j+1)* W / numBinsW;

        double L = 0.0;
        for (int h=beginH; h<endH; ++h)
          for (int w=beginW; w<endW; ++w)
          {
            size_t idx = (size_t)(h*W + w)*3;
            vec3f c{img[idx], img[idx+1], img[idx+2]};
            L += luminance(c);
          }
        L /= double((endH-beginH)*(endW-beginW));
        if (L > eps)
        {
          logSum += std::log2(L);
          count++;
        }
      }

    float exposure = (count > 0) ? (key / std::exp2(logSum / count)) : 1.f;
    float* dstPtr = getDstPtr();
    *dstPtr = exposure;
  }

OIDN_NAMESPACE_END
