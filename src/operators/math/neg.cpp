#include "origin/core/operator.h"
#include "origin/utils/log.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Neg::forward(const std::vector<Tensor> &xs)
{
    // 使用抽象层进行负号运算
    auto result = -mat(xs[0]);
    auto y      = convert_mat_to_tensor(std::move(result));
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Neg::backward(const std::vector<Tensor> &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    // 使用抽象层进行梯度计算
    auto result = -mat(gys[0]);
    auto gx     = convert_mat_to_tensor(std::move(result));
    std::vector<Tensor> gxs;
    gxs.push_back(gx);

    return gxs;
}

void Neg::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (&input1 != &kNullTensor_)
    {
        THROW_INVALID_ARG("Neg is a unary operator, cannot accept two operands");
    }

    // 原地操作：input0 = -input0
    mat(input0).neg_inplace();
}

Tensor neg(const std::vector<Tensor> &xs)
{
    return (*std::shared_ptr<Operator>(new Neg()))(xs)[0];
}

Tensor neg(const Tensor &x)
{
    auto xs = std::vector<Tensor>();
    xs.emplace_back(x);
    return neg(xs);
}

void neg_(Tensor &x)
{
    // 创建 Neg 实例并调用 forward_inplace
    Neg op;
    op.forward_inplace(x, Operator::kNullTensor_);
}

}  // namespace functional

// 运算符重载放在 origin 命名空间下
Tensor operator-(const Tensor &x)
{
    return functional::neg(x);
}

}  // namespace origin
