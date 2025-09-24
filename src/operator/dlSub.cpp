#include "base/dlException.h"
#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Sub::Forward(const NdArrayPtrList &xs)
{
    // logd("do sub");
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x0 = xs[0];
    NdArrayPtr x1 = xs[1];
    shape0_       = x0->dims();
    shape1_       = x1->dims();
    auto y        = (*x0) - (*x1);
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArrayPtrList Sub::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        DL_WARN_THROW("invalid argument size, not equal to 1");
    }

    // auto gy    = AsVariablePtr(gys[0]);
    // auto gyNeg = -gy;
    // auto gxs   = NdArrayPtrList();

    // return gxs;

    auto gx0 = AsVariablePtr(gys[0]);
    auto gx1 = -gx0;
    if (shape0_ != shape1_)
    {
        gx0 = sumTo(gx0, shape0_);
        gx1 = sumTo(gx1, shape1_);
    }
    auto gxs = NdArrayPtrList();
    gxs.push_back(AsDLArrayPtr(gx0->data_));
    gxs.push_back(AsDLArrayPtr(gx1->data_));
    return gxs;
}

VariablePtr sub(const VariablePtrList &xs)
{
    return (*std::shared_ptr<Operator>(new Sub()))(xs)[0];
}

VariablePtr sub(const VariablePtr &lhs, const VariablePtr &rhs)
{
    VariablePtrList xs = {lhs, rhs};
    return sub(xs);
}

VariablePtr operator-(const VariablePtr &lhs, const VariablePtr &rhs)
{
    return sub(lhs, rhs);
}

VariablePtr operator-(const VariablePtr &lhs, data_t rhs)
{
    auto dims = lhs->data_.dims();
    auto x    = std::make_shared<Variable>(af::constant(rhs, dims));
    return sub(lhs, x);
}
VariablePtr operator-(data_t lhs, const VariablePtr &rhs)
{
    auto dims = rhs->data_.dims();
    auto x    = std::make_shared<Variable>(af::constant(lhs, dims));
    return sub(x, rhs);
}

}  // namespace dl
