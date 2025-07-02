#include "dlFunction.h"

namespace dl
{

NdArrayPtrList Exp::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    auto o       = af::exp(*x);
    outputs.push_back(AsDLArrayPtr(o));
    return outputs;
}

NdArrayPtrList Exp::Backward(const NdArrayPtrList &gys)
{
    auto x  = this->inputs[0]->data;
    auto gx = af::exp(x) * (*gys[0]);
    return AsDLArrayPtrList(gx);
}

VariablePtr exp(const VariablePtr &x)
{
    auto f = std::shared_ptr<Function>(new Exp());
    auto y = (*f)(x);
    return y[0];
}

}  // namespace dl
