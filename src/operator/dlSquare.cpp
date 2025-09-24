#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Square::forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    auto o       = af::pow(*x, 2);
    outputs.push_back(as_dl_array_ptr(o));
    return outputs;
}

NdArrayPtrList Square::backward(const NdArrayPtrList &gys)
{
    auto x  = this->inputs_[0]->data_;
    auto gx = 2.0 * x * (*gys[0]);
    return as_dl_array_ptr_list(gx);
}

VariablePtr square(const VariablePtr &x)
{
    auto f = std::shared_ptr<Operator>(new Square());
    auto y = (*f)(x);
    return y[0];
}

}  // namespace dl
