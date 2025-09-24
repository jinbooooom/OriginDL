#include "base/dlException.h"
#include "base/dlUtils.h"
#include "dlOperator.h"

namespace dl
{

NdArrayPtrList SumTo::forward(const NdArrayPtrList &xs)
{
    auto outputs   = NdArrayPtrList();
    auto x         = *(xs[0]);
    this->x_shape_ = x.dims();
    auto y         = utils::SumTo(x, this->shape_);
    outputs.push_back(as_dl_array_ptr(y));
    return outputs;
}

NdArrayPtrList SumTo::backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        DL_WARN_THROW("invalid argument size, not equal to 1");
    }

    auto gy  = *(gys[0]);
    auto gx  = utils::BroadcastTo(gy, this->x_shape_);
    auto gxs = NdArrayPtrList();
    gxs.push_back(as_dl_array_ptr(gx));
    return gxs;
}

VariablePtr sum_to(const VariablePtr &x, const af::dim4 &shape)
{
    auto f  = std::make_shared<SumTo>(shape);
    auto ys = (*f)(x);
    return ys[0];
}

}  // namespace dl
