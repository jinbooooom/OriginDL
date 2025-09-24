#include "base/dlException.h"
#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Add::forward(const NdArrayPtrList &xs)
{
    // logd("do add");
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x0 = xs[0];
    NdArrayPtr x1 = xs[1];
    shape0_       = x0->dims();
    shape1_       = x1->dims();
    auto y        = (*x0) + (*x1);
    outputs.push_back(as_dl_array_ptr(y));

    return outputs;
}

NdArrayPtrList Add::backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        DL_WARN_THROW("invalid argument size, not equal to 1");
    }

    auto gx0 = as_variable_ptr(gys[0]);
    auto gx1 = gx0;
    if (shape0_ != shape1_)
    {
        gx0 = sum_to(gx0, shape0_);
        gx1 = sum_to(gx1, shape1_);
    }
    auto gxs = NdArrayPtrList();
    gxs.push_back(as_dl_array_ptr(gx0->data_));
    gxs.push_back(as_dl_array_ptr(gx1->data_));
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
    auto dims = lhs->data_.dims();
    auto x    = std::make_shared<Variable>(af::constant(rhs, dims));
    return add(lhs, x);
}
VariablePtr operator+(data_t lhs, const VariablePtr &rhs)
{
    auto dims = rhs->data_.dims();
    auto x    = std::make_shared<Variable>(af::constant(lhs, dims));
    return add(x, rhs);
}

}  // namespace dl
