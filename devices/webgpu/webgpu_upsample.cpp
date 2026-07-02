#include "webgpu_upsample.h"
#include "webgpu_buffer.h"

OIDN_NAMESPACE_BEGIN

  void WebGPUUpsample::submitKernels(const Ref<CancellationToken>&)
  {
    if (!src || !dst)
      throw std::logic_error("upsampling source/destination not set");

    auto srcDev = src->toDevice(engine);
    auto dstDev = dst->toDevice(engine);

    auto sb = dynamic_cast<WebGPUBuffer*>(srcDev->getBuffer());
    auto ob = dynamic_cast<WebGPUBuffer*>(dstDev->getBuffer());
    if (!sb || !ob)
      throw std::invalid_argument("tensor not on WebGPU device");

    WebGPUTensor A = {sb->getWGPUBuffer(), sb->getWGPUBufferOffset() + srcDev->getByteOffset(), 1,
                      (uint32_t)srcDesc.getC(), (uint32_t)srcDesc.getH(), (uint32_t)srcDesc.getW(), WebGPUTensorType::Input};
    WebGPUTensor O = {ob->getWGPUBuffer(), ob->getWGPUBufferOffset() + dstDev->getByteOffset(), (uint32_t)dstDesc.getC(), 1,
                      (uint32_t)dstDesc.getH(), (uint32_t)dstDesc.getW(), WebGPUTensorType::Output};

    engine->upsample2x(A, O);
  }

OIDN_NAMESPACE_END
