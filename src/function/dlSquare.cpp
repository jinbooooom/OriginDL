#include "dlFunction.h"

namespace dl
{

NdArrayPtrList Square::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    auto o       = af::pow(*x, 2);
    outputs.push_back(AsDLArrayPtr(o));
    return outputs;
}

NdArrayPtrList Square::Backward(const NdArrayPtrList &gys)
{
    auto x  = this->inputs[0]->data;
    auto gx = 2.0 * x * (*gys[0]);
    return AsDLArrayPtrList(gx);
}

VariablePtr square(const VariablePtr &x)
{
    auto f = std::shared_ptr<Function>(new Square());
    auto y = (*f)(x);
    return y[0];
}

}  // namespace dl
