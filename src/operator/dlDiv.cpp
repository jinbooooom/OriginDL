#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Div::Forward(const NdArrayPtrList &xs)
{
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x1 = xs[0];
    NdArrayPtr x2 = xs[1];
    auto y        = (*x1) / (*x2);  // 逐元素除
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArrayPtrList Div::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    auto x0 = this->inputs[0]->data;
    auto x1 = this->inputs[1]->data;  // TODO: 要判断 x1 是否为 0

    /*
        y = x0 / x1;
        dy/dx0 = 1 / x1
        dy/dx1 = -x0 / x1^2
    */
    auto gy  = *gys[0];
    auto dx0 = gy / x1;
    auto dx1 = gy * (-x0) / x1 / x1;
    auto gxs = NdArrayPtrList{AsDLArrayPtr(dx0), AsDLArrayPtr(dx1)};

    return gxs;
}

VariablePtr div(const VariablePtrList &xs)
{
    return (*std::shared_ptr<Operator>(new Div()))(xs)[0];
}

VariablePtr div(const VariablePtr &lhs, const VariablePtr &rhs)
{
    VariablePtrList xs = {lhs, rhs};
    return div(xs);
}

VariablePtr operator/(const VariablePtr &lhs, const VariablePtr &rhs)
{
    return div(lhs, rhs);
}

VariablePtr operator/(const VariablePtr &lhs, data_t rhs)
{
    auto dims = lhs->data.dims();
    auto x    = std::make_shared<Variable>(af::constant(rhs, dims));
    return div(lhs, x);
}
VariablePtr operator/(data_t lhs, const VariablePtr &rhs)
{
    auto dims = rhs->data.dims();
    auto x    = std::make_shared<Variable>(af::constant(lhs, dims));
    return div(x, rhs);
}

}  // namespace dl
