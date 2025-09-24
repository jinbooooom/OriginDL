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
    auto x  = this->inputs_[0];
    auto W  = this->inputs_[1];
    auto gy = gys[0];
    auto wt = af::transpose(W->data_);
    auto xt = af::transpose(x->data_);
    auto gx = af::matmul(*gy, wt);
    auto gw = af::matmul(xt, *gy);

    auto gxs = NdArrayPtrList();
    gxs.push_back(AsDLArrayPtr(gx));
    gxs.push_back(AsDLArrayPtr(gw));
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
