#include "origin/core/operator.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> Add::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        THROW_RUNTIME_ERROR("Add operator requires exactly 2 inputs, but got {}", xs.size());
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();

    // 使用抽象层进行加法运算
    auto result = mat(xs[0]) + mat(xs[1]);
    auto y      = convert_mat_to_tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Add::backward(const std::vector<Tensor> &gys)
{
    if (1 != gys.size())
    {
        THROW_RUNTIME_ERROR("Add backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto gx0 = gys[0];
    auto gx1 = gys[0];
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

Tensor add(const std::vector<Tensor> &xs)
{
    return (*std::shared_ptr<Operator>(new Add()))(xs)[0];
}

Tensor add(const Tensor &lhs, const Tensor &rhs)
{
    return add({lhs, rhs});
}

Tensor operator+(const Tensor &lhs, const Tensor &rhs)
{
    return add(lhs, rhs);
}

template <typename T>
Tensor operator+(const Tensor &lhs, T rhs)
{
    auto shape = lhs.shape();
    auto x     = Tensor(rhs, shape);
    return add(lhs, x);
}

template <typename T>
Tensor operator+(T lhs, const Tensor &rhs)
{
    auto shape = rhs.shape();
    auto x     = Tensor(lhs, shape);
    return add(x, rhs);
}

// 模板实例化
template Tensor operator+(const Tensor &lhs, float rhs);
template Tensor operator+(const Tensor &lhs, double rhs);
template Tensor operator+(const Tensor &lhs, int32_t rhs);
template Tensor operator+(const Tensor &lhs, int8_t rhs);
template Tensor operator+(const Tensor &lhs, unsigned long rhs);

template Tensor operator+(float lhs, const Tensor &rhs);
template Tensor operator+(double lhs, const Tensor &rhs);
template Tensor operator+(int32_t lhs, const Tensor &rhs);
template Tensor operator+(int8_t lhs, const Tensor &rhs);
template Tensor operator+(unsigned long lhs, const Tensor &rhs);

}  // namespace origin
