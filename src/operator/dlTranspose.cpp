#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Transpose::forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    auto x       = *(xs[0]);
    af::array y  = af::transpose(x);
    outputs.push_back(as_dl_array_ptr(y));
    return outputs;
}

NdArrayPtrList Transpose::backward(const NdArrayPtrList &gys)
{
    auto gy  = *(gys[0]);
    auto gx  = af::transpose(gy);
    auto gxs = NdArrayPtrList();
    gxs.push_back(as_dl_array_ptr(gx));
    return gxs;
}

VariablePtr transpose(const VariablePtr &x)
{
    auto f  = std::make_shared<Transpose>();
    auto ys = (*f)(x);
    return ys[0];
}

}  // namespace dl
