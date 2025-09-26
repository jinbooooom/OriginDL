#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Exp::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        throw std::runtime_error("Exp requires exactly 1 input");
    }

    // 使用抽象层进行指数运算
    auto result = xs[0].mat().exp();
    auto y      = Tensor(std::move(result));
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
    auto x         = &this->inputs_[0].mat();
    auto gy        = &gys[0].mat();
    auto exp_x     = x->exp();
    auto gx_result = *exp_x * *gy;
    auto gx        = Tensor(std::move(gx_result));
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

}  // namespace dl
