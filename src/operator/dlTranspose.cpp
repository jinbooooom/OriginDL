#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Transpose::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    auto x       = *(xs[0]);
    af::array y  = af::transpose(x);
    outputs.push_back(AsDLArrayPtr(y));
    return outputs;
}

NdArrayPtrList Transpose::Backward(const NdArrayPtrList &gys)
{
    auto gy  = *(gys[0]);
    auto gx  = transpose(gy);
    auto gxs = NdArrayPtrList{AsDLArrayPtr(gx)};
    return gxs;
}

VariablePtr transpose(const VariablePtr &x)
{
    auto f  = std::make_shared<Transpose>();
    auto ys = (*f)(x);
    return ys[0];
}

}  // namespace dl
