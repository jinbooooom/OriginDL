#include "origin/core/operator.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> Sub::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        THROW_RUNTIME_ERROR("Sub operator requires exactly 2 inputs, but got {}", xs.size());
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();
    // 使用抽象层进行减法运算
    auto result = mat(xs[0]) - mat(xs[1]);
    auto y      = convert_mat_to_tensor(std::move(result));
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Sub::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Sub backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto gx0 = gys[0];
    // 使用负号运算符
    auto gx1_result = -mat(gys[0]);
    auto gx1        = convert_mat_to_tensor(std::move(gx1_result));

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

Tensor sub(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Sub>();
    return (*op)(xs)[0];
}

Tensor sub(const Tensor &lhs, const Tensor &rhs)
{
    return sub({lhs, rhs});
}

Tensor operator-(const Tensor &lhs, const Tensor &rhs)
{
    return sub(lhs, rhs);
}

template <typename T>
Tensor operator-(const Tensor &lhs, T rhs)
{
    auto dims = lhs.shape();
    auto x    = Tensor(rhs, dims);
    return sub(lhs, x);
}

template <typename T>
Tensor operator-(T lhs, const Tensor &rhs)
{
    auto dims = rhs.shape();
    auto x    = Tensor(lhs, dims);
    return sub(x, rhs);
}

// 模板实例化
template Tensor operator-(const Tensor &lhs, float rhs);
template Tensor operator-(const Tensor &lhs, double rhs);
template Tensor operator-(const Tensor &lhs, int32_t rhs);
template Tensor operator-(const Tensor &lhs, int8_t rhs);
template Tensor operator-(const Tensor &lhs, unsigned long rhs);

template Tensor operator-(float lhs, const Tensor &rhs);
template Tensor operator-(double lhs, const Tensor &rhs);
template Tensor operator-(int32_t lhs, const Tensor &rhs);
template Tensor operator-(int8_t lhs, const Tensor &rhs);
template Tensor operator-(unsigned long lhs, const Tensor &rhs);

}  // namespace origin