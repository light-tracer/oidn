#include "webgpu_output_process.h"
#include "webgpu_buffer.h"
#include "core/color.h"
#include <vector>

OIDN_NAMESPACE_BEGIN

  WebGPUOutputProcess::WebGPUOutputProcess(WebGPUEngine* engine, const OutputProcessDesc& desc)
    : OutputProcess(desc), engine(engine) {}

  void WebGPUOutputProcess::submitKernels(const Ref<CancellationToken>&)
  {
    check();

    auto* sb = dynamic_cast<WebGPUBuffer*>(src->getBuffer());
    if (!sb)
      throw std::invalid_argument("source tensor not on WebGPU device");

    const int H = srcDesc.getH();
    const int W = srcDesc.getW();
    const int C = srcDesc.getC();

    std::vector<float> host(size_t(C)*H*W);
    sb->read(src->getByteOffset(), host.size()*sizeof(float), host.data());

    if (dst->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");
    float* dstPtr = static_cast<float*>(dst->getPtr());
    for (int h=0; h<H; ++h)
      for (int w=0; w<W; ++w)
        for (int c=0; c<C; ++c)
          dstPtr[(h*W + w)*3 + c] = host[(h*W + w)*C + c];
  }

OIDN_NAMESPACE_END
