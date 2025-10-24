#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"
#include "origin/mat/type_promotion.h"

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

    // 使用统一的类型提升工具
    if (TypePromotion::needs_promotion(xs))
    {
        auto promoted_tensors = TypePromotion::promote_tensors(xs);
        auto result = mat(promoted_tensors[0]) + mat(promoted_tensors[1]);
        auto y = convert_mat_to_tensor(std::move(result));
        return std::vector<Tensor>{y};
    }

    // 类型匹配，直接运算
    auto result = mat(xs[0]) + mat(xs[1]);
    auto y = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{y};
}

std::vector<Tensor> Add::backward(const std::vector<Tensor> &gys)
{
    if (1 != gys.size())
    {
        THROW_RUNTIME_ERROR("Add backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // 使用统一的类型提升工具
    if (TypePromotion::needs_promotion(this->inputs_))
    {
        auto promoted_inputs = TypePromotion::promote_tensors(this->inputs_);
        // 使用提升后的输入进行梯度计算
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
        return std::vector<Tensor>{gx0, gx1};
    }

    // 类型匹配，直接处理
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
    return std::vector<Tensor>{gx0, gx1};
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

Tensor operator+(const Tensor &lhs, const Scalar &rhs)
{
    auto x = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    return add(lhs, x);
}

Tensor operator+(const Scalar &lhs, const Tensor &rhs)
{
    return rhs + lhs;
}

}  // namespace origin
