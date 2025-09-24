#include "dlOperator.h"

namespace dl
{

NdArrayPtrList MatMul::forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    NdArrayPtr w = xs[1];
    auto y       = af::matmul(*x, *w);
    outputs.push_back(as_dl_array_ptr(y));

    return outputs;
}

NdArrayPtrList MatMul::backward(const NdArrayPtrList &gys)
{
    auto x  = this->inputs_[0];
    auto W  = this->inputs_[1];
    auto gy = gys[0];
    auto wt = af::transpose(W->data_);
    auto xt = af::transpose(x->data_);
    auto gx = af::matmul(*gy, wt);
    auto gw = af::matmul(xt, *gy);

    auto gxs = NdArrayPtrList();
    gxs.push_back(as_dl_array_ptr(gx));
    gxs.push_back(as_dl_array_ptr(gw));
    return gxs;
}

VariablePtr mat_mul(const VariablePtr &x, const VariablePtr &w)
{
    auto f               = std::make_shared<MatMul>();
    VariablePtrList args = {x, w};
    auto ys              = (*f)(args);
    return ys[0];
}

}  // namespace dl
