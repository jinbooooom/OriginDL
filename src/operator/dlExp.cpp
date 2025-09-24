#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Exp::forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    auto o       = af::exp(*x);
    outputs.push_back(as_dl_array_ptr(o));
    return outputs;
}

NdArrayPtrList Exp::backward(const NdArrayPtrList &gys)
{
    auto x  = this->inputs_[0]->data_;
    auto gx = af::exp(x) * (*gys[0]);
    return as_dl_array_ptr_list(gx);
}

VariablePtr exp(const VariablePtr &x)
{
    auto f = std::shared_ptr<Operator>(new Exp());
    auto y = (*f)(x);
    return y[0];
}

}  // namespace dl
