#include "origin/core/operator.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Exp::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Exp operator requires exactly 1 input, but got {}", xs.size());
    }

    // 使用抽象层进行指数运算
    auto result = mat(xs[0]).exp();
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Exp::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Exp backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // 使用抽象层进行梯度计算
    auto &x        = mat(this->inputs_[0]);
    auto &gy       = mat(gys[0]);
    auto exp_x     = x.exp();
    auto gx_result = *exp_x * gy;
    auto gx        = convert_mat_to_tensor(std::move(gx_result));
    return std::vector<Tensor>{std::move(gx)};
}

void Exp::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (&input1 != &kNullTensor_)
    {
        THROW_INVALID_ARG("Exp is a unary operator, cannot accept two operands");
    }

    // 原地操作：input0 = exp(input0)
    mat(input0).exp_inplace();
}

Tensor exp(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Exp>();
    return (*op)(xs)[0];
}

Tensor exp(const Tensor &x)
{
    auto xs = std::vector<Tensor>();
    xs.emplace_back(x);
    return exp(xs);
}

void exp_(Tensor &x)
{
    // 创建 Exp 实例并调用 forward_inplace
    Exp op;
    op.forward_inplace(x, Operator::kNullTensor_);
}

}  // namespace functional
}  // namespace origin
