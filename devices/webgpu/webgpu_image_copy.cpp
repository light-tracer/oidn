#include "webgpu_image_copy.h"
#include <cstring>

OIDN_NAMESPACE_BEGIN

  WebGPUImageCopy::WebGPUImageCopy(WebGPUEngine* engine)
    : engine(engine) {}

  void WebGPUImageCopy::submitKernels(const Ref<CancellationToken>&)
  {
    check();
    std::memcpy(dst->getPtr(), src->getPtr(), src->getDesc().getByteSize());
  }

OIDN_NAMESPACE_END
