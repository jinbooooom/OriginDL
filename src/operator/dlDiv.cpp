#include "base/dlException.h"
#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Div::Forward(const NdArrayPtrList &xs)
{
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x0 = xs[0];
    NdArrayPtr x1 = xs[1];
    shape0_       = x0->dims();
    shape1_       = x1->dims();
    auto y        = (*x0) / (*x1);  // 逐元素除
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArrayPtrList Div::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        DL_WARN_THROW("invalid argument size, not equal to 1");
    }

    auto x0 = this->inputs_[0]->data_;
    auto x1 = this->inputs_[1]->data_;  // TODO: 要判断 x1 是否为 0

    /*
        y = x0 / x1;
        dy/dx0 = 1 / x1
        dy/dx1 = -x0 / x1^2
    */
    auto gy  = *gys[0];
    auto dx0 = gy / x1;
    auto dx1 = gy * (-x0) / x1 / x1;

    VariablePtr dx0_ = AsVariablePtr(AsDLArrayPtr(dx0));
    VariablePtr dx1_ = AsVariablePtr(AsDLArrayPtr(dx1));
    if (shape0_ != shape1_)
    {
        dx0_ = sumTo(AsVariablePtr(AsDLArrayPtr(dx0)), shape0_);
        dx1_ = sumTo(AsVariablePtr(AsDLArrayPtr(dx1)), shape1_);
    }

    auto gxs = NdArrayPtrList();
    gxs.push_back(AsDLArrayPtr(dx0_->data_));
    gxs.push_back(AsDLArrayPtr(dx1_->data_));
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
    auto dims = lhs->data_.dims();
    auto x    = std::make_shared<Variable>(af::constant(rhs, dims));
    return div(lhs, x);
}
VariablePtr operator/(data_t lhs, const VariablePtr &rhs)
{
    auto dims = rhs->data_.dims();
    auto x    = std::make_shared<Variable>(af::constant(lhs, dims));
    return div(x, rhs);
}

}  // namespace dl
