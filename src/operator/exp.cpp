#include "origin/core/operator.h"

namespace origin
{

std::vector<Tensor> Exp::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        throw std::runtime_error("Exp requires exactly 1 input");
    }

    // 使用抽象层进行指数运算
    auto result = mat(xs[0]).exp();
    auto y      = convert_mat_to_tensor(std::move(result));
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Exp::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("Exp backward requires exactly 1 gradient");
    }

    // 使用抽象层进行梯度计算
    auto x         = &mat(this->inputs_[0]);
    auto gy        = &mat(gys[0]);
    auto exp_x     = x->exp();
    auto gx_result = *exp_x * *gy;
    auto gx        = convert_mat_to_tensor(std::move(gx_result));
    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
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

}  // namespace origin
