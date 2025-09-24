#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Reshape::forward(const NdArrayPtrList &xs)
{
    auto outputs   = NdArrayPtrList();
    auto x         = *(xs[0]);
    this->x_shape_ = x.dims();
    auto y         = af::moddims(x, this->shape_);
    outputs.push_back(as_dl_array_ptr(y));
    return outputs;
}

NdArrayPtrList Reshape::backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    auto gy  = *(gys[0]);
    auto gx  = af::moddims(gy, this->x_shape_);
    auto gxs = NdArrayPtrList();
    gxs.push_back(as_dl_array_ptr(gx));
    return gxs;
}

VariablePtr reshape(const VariablePtr &x, const af::dim4 shape)
{
    // TODO：如果 shape 相同就什么都不做

    auto f  = std::make_shared<Reshape>(shape);
    auto ys = (*f)(x);
    return ys[0];
}

}  // namespace dl
