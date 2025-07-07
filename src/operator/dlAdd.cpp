#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Add::Forward(const NdArrayPtrList &xs)
{
    // logd("do add");
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x1 = xs[0];
    NdArrayPtr x2 = xs[1];
    auto y        = (*x1) + (*x2);
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArrayPtrList Add::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    auto gxs = NdArrayPtrList{gys[0], gys[0]};

    return gxs;
}

VariablePtr add(const VariablePtrList &xs)
{
    return (*std::shared_ptr<Operator>(new Add()))(xs)[0];
}

VariablePtr add(const VariablePtr &lhs, const VariablePtr &rhs)
{
    VariablePtrList xs = {lhs, rhs};
    return add(xs);
}

VariablePtr operator+(const VariablePtr &lhs, const VariablePtr &rhs)
{
    return add(lhs, rhs);
}

VariablePtr operator+(const VariablePtr &lhs, data_t rhs)
{
    auto dims = lhs->data.dims();
    auto x    = std::make_shared<Variable>(af::constant(rhs, dims));
    return add(lhs, x);
}
VariablePtr operator+(data_t lhs, const VariablePtr &rhs)
{
    auto dims = rhs->data.dims();
    auto x    = std::make_shared<Variable>(af::constant(lhs, dims));
    return add(x, rhs);
}

}  // namespace dl
