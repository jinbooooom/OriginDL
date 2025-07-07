#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Neg::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArray x0   = -(*xs[0]);
    outputs.push_back(AsDLArrayPtr(x0));
    return outputs;
}

NdArrayPtrList Neg::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    auto gy  = -(*gys[0]);
    auto gxs = NdArrayPtrList{AsDLArrayPtr(gy)};

    return gxs;
}

VariablePtr neg(const VariablePtrList &xs)
{
    return (*std::shared_ptr<Operator>(new Neg()))(xs)[0];
}

VariablePtr neg(const VariablePtr &x)
{
    auto xs = VariablePtrList();
    xs.emplace_back(x);
    return neg(xs);
}

VariablePtr operator-(const VariablePtr &x)
{
    return neg(x);
}

}  // namespace dl
