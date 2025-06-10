#include "webgpu_input_process.h"
#include "webgpu_buffer.h"
#include "core/color.h"
#include <vector>

OIDN_NAMESPACE_BEGIN

  WebGPUInputProcess::WebGPUInputProcess(WebGPUEngine* engine, const InputProcessDesc& desc)
    : InputProcess(engine, desc), engine(engine) {}

  void WebGPUInputProcess::submitKernels(const Ref<CancellationToken>&)
  {
    check();

    if (!color || albedo || normal)
      throw std::logic_error("unsupported input process configuration");
    if (color->getFormat() != Format::Float3)
      throw std::invalid_argument("unsupported image format");

    const int H = dstDesc.getH();
    const int W = dstDesc.getW();
    const int C = dstDesc.getC();

    std::vector<float> host(size_t(C)*H*W);
    const float* srcPtr = static_cast<const float*>(color->getPtr());
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        for (int c = 0; c < C; ++c)
          host[(h*W + w)*C + c] = srcPtr[(h*W + w)*3 + c];

    auto* buf = dynamic_cast<WebGPUBuffer*>(dst->getBuffer());
    if (!buf)
      throw std::invalid_argument("destination tensor not on WebGPU device");

    buf->write(dst->getByteOffset(), host.size()*sizeof(float), host.data());
  }

OIDN_NAMESPACE_END
