#include "dlOperator.h"

namespace dl
{

NdArrayPtrList MatMul::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    NdArrayPtr w = xs[1];
    auto y       = af::matmul(*x, *w);
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArrayPtrList MatMul::Backward(const NdArrayPtrList &gys)
{
    auto x  = this->inputs[0];
    auto W  = this->inputs[1];
    auto gy = gys[0];
    auto wt = af::transpose(W->mData);
    auto xt = af::transpose(x->mData);
    auto gx = af::matmul(*gy, wt);
    auto gw = af::matmul(xt, *gy);

    auto gxs = NdArrayPtrList{AsDLArrayPtr(gx), AsDLArrayPtr(gw)};
    return gxs;
}

VariablePtr matMul(const VariablePtr &x, const VariablePtr &W)
{
    auto f               = std::make_shared<MatMul>();
    VariablePtrList args = {x, W};
    auto ys              = (*f)(args);
    return ys[0];
}

}  // namespace dl
