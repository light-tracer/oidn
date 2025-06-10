#include "webgpu_pool.h"
#include "webgpu_buffer.h"

OIDN_NAMESPACE_BEGIN

  void WebGPUPool::submitKernels(const Ref<CancellationToken>&)
  {
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    auto srcDev = src->toDevice(engine);
    auto dstDev = dst->toDevice(engine);

    auto sb = dynamic_cast<WebGPUBuffer*>(srcDev->getBuffer());
    auto ob = dynamic_cast<WebGPUBuffer*>(dstDev->getBuffer());
    if (!sb || !ob)
      throw std::invalid_argument("tensor not on WebGPU device");

    WebGPUTensor A = {sb->getWGPUBuffer(), srcDev->getByteOffset(), 1,
                      (uint32_t)srcDesc.getC(), (uint32_t)srcDesc.getH(), (uint32_t)srcDesc.getW(), WebGPUTensorType::INPUT};
    WebGPUTensor O = {ob->getWGPUBuffer(), dstDev->getByteOffset(), (uint32_t)dstDesc.getC(), 1,
                      (uint32_t)dstDesc.getH(), (uint32_t)dstDesc.getW(), WebGPUTensorType::OUTPUT};

    engine->pool2x2(A, O);
  }

OIDN_NAMESPACE_END
