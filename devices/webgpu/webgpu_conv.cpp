#include "webgpu_conv.h"
#include "webgpu_buffer.h"

OIDN_NAMESPACE_BEGIN

  WebGPUConv::WebGPUConv(WebGPUEngine* engine, const ConvDesc& desc)
    : Conv(desc), engine(engine)
  {
    // The base Conv constructor computes dstDesc (same-padding convolution,
    // with H/W halved or doubled by the fused Pool/Upsample post-op)
    if (weightDesc.getH() % 2 == 0 || weightDesc.getW() % 2 == 0)
      throw std::invalid_argument("WebGPU convolution supports only odd kernel sizes");
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

    WebGPUConvParams params;
    params.padH = (weightDesc.getH() - 1) / 2;
    params.padW = (weightDesc.getW() - 1) / 2;
    params.activation = (activation == Activation::ReLU) ? 1 : 0;

    switch (postOp)
    {
    case PostOp::None:     params.postOp = 0; break;
    case PostOp::Pool:     params.postOp = 1; break;
    case PostOp::Upsample: params.postOp = 2; break;
    default:
      throw std::invalid_argument("unsupported convolution postop");
    }

    WebGPUTensor A  = {sb->getWGPUBuffer(), sb->getWGPUBufferOffset() + srcDev->getByteOffset(), 1,
                       (uint32_t)srcDesc.getC(), (uint32_t)srcDesc.getH(), (uint32_t)srcDesc.getW(), WebGPUTensorType::Input};
    WebGPUTensor Wt = {wb->getWGPUBuffer(), wb->getWGPUBufferOffset() + weightDev->getByteOffset(), (uint32_t)weightDesc.getO(),
                       (uint32_t)weightDesc.getI(), (uint32_t)weightDesc.getH(), (uint32_t)weightDesc.getW(), WebGPUTensorType::Const};
    WebGPUTensor O  = {ob->getWGPUBuffer(), ob->getWGPUBufferOffset() + dstDev->getByteOffset(), 1,
                       (uint32_t)dstDesc.getC(), (uint32_t)dstDesc.getH(), (uint32_t)dstDesc.getW(), WebGPUTensorType::Output};

    WebGPUTensor B;
    if (biasDesc.getRank() == 1)
    {
      params.biasMode = 0;
      B = {bb->getWGPUBuffer(), bb->getWGPUBufferOffset() + biasDev->getByteOffset(), (uint32_t)biasDesc.getX(), 1, 1, 1, WebGPUTensorType::Const};
    }
    else // rank 3: full-tensor bias with the same shape as dst
    {
      if (postOp != PostOp::None)
        throw std::invalid_argument("tensor bias requires PostOp::None");

      if (bb->getWGPUBuffer() == ob->getWGPUBuffer() &&
          bb->getWGPUBufferOffset() + biasDev->getByteOffset() ==
          ob->getWGPUBufferOffset() + dstDev->getByteOffset())
      {
        // dst doubles as bias (ConcatConvHWC): read the bias through the dst
        // binding to respect WebGPU buffer aliasing rules, and bind a dummy
        // buffer in the bias slot
        params.biasMode = 2;
        B = {engine->getDummyBuffer(), 0, 1, 1, 1, 1, WebGPUTensorType::Const};
      }
      else
      {
        params.biasMode = 1;
        B = {bb->getWGPUBuffer(), bb->getWGPUBufferOffset() + biasDev->getByteOffset(), 1,
             (uint32_t)biasDesc.getC(), (uint32_t)biasDesc.getH(), (uint32_t)biasDesc.getW(), WebGPUTensorType::Const};
      }
    }

    engine->conv2d(A, Wt, B, O, params);
  }

OIDN_NAMESPACE_END
