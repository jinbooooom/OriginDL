#include "base/dlUtils.h"
#include "dlOperator.h"

namespace dl
{

NdArrayPtrList SumTo::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    auto x       = *(xs[0]);
    this->xShape = x.dims();
    auto y       = utils::SumTo(x, this->shape);
    outputs.push_back(AsDLArrayPtr(y));
    return outputs;
}

NdArrayPtrList SumTo::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    auto gy  = *(gys[0]);
    auto gx  = utils::BroadcastTo(gy, this->xShape);
    auto gxs = NdArrayPtrList{AsDLArrayPtr(gx)};
    return gxs;
}

VariablePtr sumTo(const VariablePtr &x, const af::dim4 &shape)
{
    auto f  = std::make_shared<SumTo>(shape);
    auto ys = (*f)(x);
    return ys[0];
}

}  // namespace dl
