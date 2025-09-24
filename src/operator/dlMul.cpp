#include "base/dlException.h"
#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Mul::Forward(const NdArrayPtrList &xs)
{
    // logd("do Mul");
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x0 = xs[0];
    NdArrayPtr x1 = xs[1];
    shape0_       = x0->dims();
    shape1_       = x1->dims();
    auto y        = (*x0) * (*x1);  // 逐元素乘
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArrayPtrList Mul::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        DL_WARN_THROW("invalid argument size, not equal to 1");
    }

    auto x0  = this->inputs_[0]->data_;
    auto x1  = this->inputs_[1]->data_;
    auto dx0 = AsDLArrayPtr((*gys[0]) * (x1));
    auto dx1 = AsDLArrayPtr((*gys[0]) * (x0));

    VariablePtr dx0_ = AsVariablePtr(dx0);
    VariablePtr dx1_ = AsVariablePtr(dx1);
    if (shape0_ != shape1_)
    {
        dx0_ = sumTo(AsVariablePtr(dx0), shape0_);
        dx1_ = sumTo(AsVariablePtr(dx1), shape1_);
    }
    auto gxs = NdArrayPtrList();
    gxs.push_back(AsDLArrayPtr(dx0_->data_));
    gxs.push_back(AsDLArrayPtr(dx1_->data_));
    return gxs;
}

VariablePtr mul(const VariablePtrList &xs)
{
    return (*std::shared_ptr<Operator>(new Mul()))(xs)[0];
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
    auto dims = lhs->data_.dims();
    auto x    = std::make_shared<Variable>(af::constant(rhs, dims));
    return mul(lhs, x);
}
VariablePtr operator*(data_t lhs, const VariablePtr &rhs)
{
    auto dims = rhs->data_.dims();
    auto x    = std::make_shared<Variable>(af::constant(lhs, dims));
    return mul(x, rhs);
}

}  // namespace dl
