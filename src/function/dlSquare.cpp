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

VariablePtr operator^(const VariablePtr &lhs, int n)
{
    if (n < 0)
    {
        loge("Exponent n{} must be non-negative", n);
    }

    // 要考虑 0 的 0次方在数学上无定义的情况
    // if (lhs.data == 0 && n == 0) {
    //     throw std::domain_error("0^0 is undefined");
    // }

    if (2 == n)
    {
        return square(lhs);
    }
    else
    {
        loge("OriginDL not support n != 2");
        exit(0);
    }
}

}  // namespace dl
