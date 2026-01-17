#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/type_promotion.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
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
        auto result           = mat(promoted_tensors[0]) + mat(promoted_tensors[1]);
        auto y                = convert_mat_to_tensor(std::move(result));
        return std::vector<Tensor>{std::move(y)};
    }

    // 类型匹配，直接运算
    auto result = mat(xs[0]) + mat(xs[1]);

    auto y = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
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
                gx0 = functional::sum_to(gx0, shape0_);
            }
            if (gx1.shape() != shape1_)
            {
                gx1 = functional::sum_to(gx1, shape1_);
            }
        }
        return std::vector<Tensor>{std::move(gx0), std::move(gx1)};
    }

    // 类型匹配，直接处理
    auto gx0 = gys[0];
    auto gx1 = gys[0];
    if (shape0_ != shape1_)
    {
        // 实现 sum_to 功能：将梯度广播回原始形状
        if (gx0.shape() != shape0_)
        {
            gx0 = functional::sum_to(gx0, shape0_);
        }
        if (gx1.shape() != shape1_)
        {
            gx1 = functional::sum_to(gx1, shape1_);
        }
    }
    return std::vector<Tensor>{std::move(gx0), std::move(gx1)};
}

void Add::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (&input1 == &kNullTensor_)
    {
        THROW_INVALID_ARG("Add requires two operands, cannot be used as unary operator");
    }

    // 原地操作：input0 = input0 + input1
    if (TypePromotion::needs_promotion({input0, input1}))
    {
        auto promoted_tensors = TypePromotion::promote_tensors({input0, input1});
        // 如果input0需要类型提升，需要先转换
        if (input0.dtype() != promoted_tensors[0].dtype())
        {
            input0 = input0.to(promoted_tensors[0].dtype());
        }
        // 如果input1需要类型提升，需要先转换
        Tensor input1_promoted =
            (input1.dtype() != promoted_tensors[1].dtype()) ? input1.to(promoted_tensors[1].dtype()) : input1;

        // 使用 mat() 方法获取 Mat 引用并执行原地操作
        mat(input0).add_inplace(mat(input1_promoted));
    }
    else
    {
        // 类型匹配，直接执行原地操作
        mat(input0).add_inplace(mat(input1));
    }
}

Tensor add(const std::vector<Tensor> &xs)
{
    return (*std::shared_ptr<Operator>(new Add()))(xs)[0];
}

Tensor add(const Tensor &lhs, const Tensor &rhs)
{
    return add({lhs, rhs});
}

void add_(Tensor &lhs, const Tensor &rhs)
{
    // 创建 Add 实例并调用 forward_inplace
    Add op;
    op.forward_inplace(lhs, rhs);
}

}  // namespace functional

// 运算符重载放在 origin 命名空间下
Tensor operator+(const Tensor &lhs, const Tensor &rhs)
{
    return functional::add(lhs, rhs);
}

Tensor operator+(const Tensor &lhs, const Scalar &rhs)
{
    auto x = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    return functional::add(lhs, x);
}

Tensor operator+(const Scalar &lhs, const Tensor &rhs)
{
    return rhs + lhs;
}

}  // namespace origin
