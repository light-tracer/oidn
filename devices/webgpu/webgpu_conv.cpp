#include "webgpu_conv.h"
#include "webgpu_buffer.h"

OIDN_NAMESPACE_BEGIN

  WebGPUConv::WebGPUConv(WebGPUEngine* engine, const ConvDesc& desc)
    : Conv(desc), engine(engine)
  {
    const int oh = srcDesc.getH() - weightDesc.getH() + 1;
    const int ow = srcDesc.getW() - weightDesc.getW() + 1;
    TensorDims dstDims{weightDesc.getO(), oh, ow};
    TensorDims dstPaddedDims{weightDesc.getPaddedO(), oh, ow};
    dstDesc = {dstDims, dstPaddedDims, srcDesc.layout, srcDesc.dataType};
  }

  void WebGPUConv::submitKernels(const Ref<CancellationToken>&)
  {
    if (!src || !weight || !bias || !dst)
      throw std::logic_error("convolution parameters not set");

    auto srcDev    = src->toDevice(engine);
    auto weightDev = weight->toDevice(engine);
    auto biasDev   = bias->toDevice(engine);
    auto dstDev    = dst->toDevice(engine);

    auto sb = dynamic_cast<WebGPUBuffer*>(srcDev->getBuffer());
    auto wb = dynamic_cast<WebGPUBuffer*>(weightDev->getBuffer());
    auto bb = dynamic_cast<WebGPUBuffer*>(biasDev->getBuffer());
    auto ob = dynamic_cast<WebGPUBuffer*>(dstDev->getBuffer());
    if (!sb || !wb || !bb || !ob)
      throw std::invalid_argument("tensor not on WebGPU device");

    WebGPUTensor A  = {sb->getWGPUBuffer(), srcDev->getByteOffset(), 1,
                       (uint32_t)srcDesc.getC(), (uint32_t)srcDesc.getH(), (uint32_t)srcDesc.getW(), WebGPUTensorType::INPUT};
    WebGPUTensor Wt = {wb->getWGPUBuffer(), weightDev->getByteOffset(), (uint32_t)weightDesc.getO(),
                       (uint32_t)weightDesc.getI(), (uint32_t)weightDesc.getH(), (uint32_t)weightDesc.getW(), WebGPUTensorType::CONST};
    WebGPUTensor B  = {bb->getWGPUBuffer(), biasDev->getByteOffset(), (uint32_t)biasDesc.getX(), 1,1,1, WebGPUTensorType::CONST};
    WebGPUTensor O  = {ob->getWGPUBuffer(), dstDev->getByteOffset(), (uint32_t)dstDesc.getC(), 1,
                       (uint32_t)dstDesc.getH(), (uint32_t)dstDesc.getW(), WebGPUTensorType::OUTPUT};

    engine->conv2d_eltwise(A, Wt, B, O);
  }

OIDN_NAMESPACE_END
