#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Mul::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        throw std::runtime_error("Mul requires exactly 2 inputs");
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();

    // 使用抽象层进行乘法运算
    auto result = xs[0].mat() * xs[1].mat();
    auto y      = Tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Mul::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("Mul backward requires exactly 1 gradient");
    }

    auto x0 = &this->inputs_[0].mat();
    auto x1 = &this->inputs_[1].mat();
    auto gy = &gys[0].mat();

    // 使用抽象层进行梯度计算
    auto gx0_result = *gy * *x1;
    auto gx1_result = *gy * *x0;

    auto gx0 = Tensor(std::move(gx0_result));
    auto gx1 = Tensor(std::move(gx1_result));

    if (shape0_ != shape1_)
    {
        // 实现 sum_to 功能：将梯度广播回原始形状
        if (gx0.shape() != shape0_)
        {
            gx0 = sum_to(gx0, shape0_);
        }
        if (gx1.shape() != shape1_)
        {
            gx1 = sum_to(gx1, shape1_);
        }
    }

    std::vector<Tensor> gxs;
    gxs.push_back(gx0);
    gxs.push_back(gx1);
    return gxs;
}

Tensor mul(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Mul>();
    return (*op)(xs)[0];
}

Tensor mul(const Tensor &lhs, const Tensor &rhs)
{
    return mul({lhs, rhs});
}

Tensor operator*(const Tensor &lhs, const Tensor &rhs)
{
    return mul(lhs, rhs);
}

Tensor operator*(const Tensor &lhs, data_t rhs)
{
    auto shape = lhs.shape();
    auto x     = Tensor::constant(rhs, shape);
    return mul(lhs, x);
}

Tensor operator*(data_t lhs, const Tensor &rhs)
{
    auto shape = rhs.shape();
    auto x     = Tensor::constant(lhs, shape);
    return mul(x, rhs);
}

}  // namespace dl