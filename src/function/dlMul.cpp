#include "dlFunction.h"

namespace dl
{

NdArrayPtrList Mul::Forward(const NdArrayPtrList &xs)
{
    // logd("do Mul");
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x1 = xs[0];
    NdArrayPtr x2 = xs[1];
    auto y        = (*x1) * (*x2);  // 逐元素乘
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArrayPtrList Mul::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    auto x0  = this->inputs[0]->data;
    auto x1  = this->inputs[1]->data;
    auto dx0 = AsDLArrayPtr((*gys[0]) * (x1));
    auto dx1 = AsDLArrayPtr((*gys[0]) * (x0));
    auto gxs = NdArrayPtrList{dx0, dx1};

    return gxs;
}

VariablePtr mul(const VariablePtrList &xs)
{
    return (*std::shared_ptr<Function>(new Mul()))(xs)[0];
}

VariablePtr mul(const VariablePtr &lhs, const VariablePtr &rhs)
{
    VariablePtrList xs = {lhs, rhs};
    return mul(xs);
}

VariablePtr operator*(const VariablePtr &lhs, const VariablePtr &rhs)
{
    return mul(lhs, rhs);
}

VariablePtr operator*(const VariablePtr &lhs, data_t rhs)
{
    auto dims = lhs->data.dims();
    auto x    = std::make_shared<Variable>(af::constant(rhs, dims));
    return mul(lhs, x);
}
VariablePtr operator*(data_t lhs, const VariablePtr &rhs)
{
    auto dims = rhs->data.dims();
    auto x    = std::make_shared<Variable>(af::constant(lhs, dims));
    return mul(x, rhs);
}

}  // namespace dl
