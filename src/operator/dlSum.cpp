#include "base/dlUtils.h"
#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Sum::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    auto x       = *(xs[0]);
    this->xShape = x.dims();

    NdArray y;
    if (-1 == axis)
    {
        auto n = x.numdims();
        logd("numdims of matrix x:", n);
        y = x;
        for (unsigned i = 0; i < n; ++i)
        {
            y = af::sum(y);
        }
        // y 的维度 [1, 1, 1, 1]
    }
    else
    {
        y = af::sum(x, axis);
    }
    outputs.push_back(AsDLArrayPtr(y));
    return outputs;
}

NdArrayPtrList Sum::Backward(const NdArrayPtrList &gys)
{
    auto gy  = *(gys[0]);
    auto gx  = utils::BroadcastTo(gy, xShape);
    auto gxs = NdArrayPtrList{AsDLArrayPtr(gx)};
    return gxs;
}

VariablePtr sum(const VariablePtr &x, int axis)  // -1 意味着所有元素相加
{
    auto f  = std::make_shared<Sum>();
    auto ys = (*f)(x);
    return ys[0];
}

}  // namespace dl
