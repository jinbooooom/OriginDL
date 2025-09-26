#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> MatMul::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        throw std::runtime_error("MatMul requires exactly 2 inputs");
    }

    // 使用抽象层进行矩阵乘法运算
    auto result = xs[0].mat() * xs[1].mat();
    auto y      = Tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> MatMul::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("MatMul backward requires exactly 1 gradient");
    }

    auto x  = &this->inputs_[0].mat();
    auto w  = &this->inputs_[1].mat();
    auto gy = &gys[0].mat();

    // 使用抽象层进行梯度计算
    auto w_T = w->transpose();
    auto x_T = x->transpose();

    auto gx_result = *gy * *w_T;
    auto gw_result = *x_T * *gy;

    auto gx = Tensor(std::move(gx_result));
    auto gw = Tensor(std::move(gw_result));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    outputs.push_back(gw);
    return outputs;
}

Tensor matmul(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<MatMul>();
    return (*op)(xs)[0];
}

Tensor matmul(const Tensor &lhs, const Tensor &rhs)
{
    return matmul({lhs, rhs});
}

Tensor mat_mul(const Tensor &lhs, const Tensor &rhs)
{
    return matmul(lhs, rhs);
}

}  // namespace dl