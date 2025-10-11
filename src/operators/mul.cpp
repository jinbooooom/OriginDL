#include "origin/core/operator.h"

namespace origin
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
    auto result = mat(xs[0]) * mat(xs[1]);
    auto y      = convert_mat_to_tensor(std::move(result));

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

    auto x0 = &mat(this->inputs_[0]);
    auto x1 = &mat(this->inputs_[1]);
    auto gy = &mat(gys[0]);

    // 使用抽象层进行梯度计算
    auto gx0_result = *gy * *x1;
    auto gx1_result = *gy * *x0;

    auto gx0 = convert_mat_to_tensor(std::move(gx0_result));
    auto gx1 = convert_mat_to_tensor(std::move(gx1_result));

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

template <typename T>
Tensor operator*(const Tensor &lhs, T rhs)
{
    auto shape = lhs.shape();
    auto x     = Tensor(rhs, shape);
    return mul(lhs, x);
}

template <typename T>
Tensor operator*(T lhs, const Tensor &rhs)
{
    auto shape = rhs.shape();
    auto x     = Tensor(lhs, shape);
    return mul(x, rhs);
}

// 模板实例化
template Tensor operator*(const Tensor &lhs, float rhs);
template Tensor operator*(const Tensor &lhs, double rhs);
template Tensor operator*(const Tensor &lhs, int32_t rhs);
template Tensor operator*(const Tensor &lhs, int8_t rhs);
template Tensor operator*(const Tensor &lhs, unsigned long rhs);

template Tensor operator*(float lhs, const Tensor &rhs);
template Tensor operator*(double lhs, const Tensor &rhs);
template Tensor operator*(int32_t lhs, const Tensor &rhs);
template Tensor operator*(int8_t lhs, const Tensor &rhs);
template Tensor operator*(unsigned long lhs, const Tensor &rhs);

}  // namespace origin