#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/type_promotion.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Div::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("Div operator requires exactly 2 inputs, but got {}", xs.size());
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();

    auto [x0, x1] = TypePromotion::promote_tensors_maybe_owned(xs[0], xs[1]);
    auto result   = mat(x0) / mat(x1);
    auto y        = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Div::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(1 != gys.size()))
    {
        THROW_RUNTIME_ERROR("Div backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // Div的梯度计算：
    // 对于 y = x0 / x1：
    // ∂y/∂x0 = 1/x1
    // ∂y/∂x1 = -x0/x1²
    // 需要使用提升后的输入进行梯度计算
    auto [x0, x1] = TypePromotion::promote_tensors_maybe_owned(this->inputs_[0], this->inputs_[1]);
    auto &gy      = mat(gys[0]);

    // ∂y/∂x0 = gy * (1/x1) = gy / x1
    auto &x0_mat    = mat(x0);
    auto &x1_mat    = mat(x1);
    auto gx0_result = gy / x1_mat;
    auto gx0        = convert_mat_to_tensor(std::move(gx0_result));

    // ∂y/∂x1 = gy * (-x0/x1²) = -gy * x0 / x1²
    auto x1_squared = x1_mat * x1_mat;
    auto temp_mult  = gy * x0_mat;
    auto temp_div   = *temp_mult / *x1_squared;
    auto gx1_result = -*temp_div;
    auto gx1        = convert_mat_to_tensor(std::move(gx1_result));

    // 统一处理形状广播：将梯度广播回原始形状
    if (shape0_ != shape1_)
    {
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

void Div::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (unlikely(&input1 == &kNullTensor_))
    {
        THROW_INVALID_ARG("Div requires two operands, cannot be used as unary operator");
    }

    // 原地操作：input0 = input0 / input1
    // 统一处理：无论是否需要类型提升，都使用相同的逻辑
    DataType promoted_type = TypePromotion::promote_types(input0.dtype(), input1.dtype());

    // 因为 input0 需要原地修改，所以不用临时的 MaybeOwned<Tensor>，而是直接修改 input0
    if (input0.dtype() != promoted_type)
    {
        input0 = input0.to(promoted_type);
    }

    // input1 使用 MaybeOwned 优化：类型匹配时借用，不匹配时创建新的 Tensor 并拥有所有权
    auto x1_maybe = TypePromotion::to_type_maybe_owned(input1, promoted_type);

    // 使用 mat() 方法获取 Mat 引用并执行原地操作
    mat(input0).div_inplace(mat(x1_maybe));
}

Tensor div(const std::vector<Tensor> &xs)
{
    return (*std::shared_ptr<Operator>(new Div()))(xs)[0];
}

Tensor div(const Tensor &lhs, const Tensor &rhs)
{
    return div({lhs, rhs});
}

void div_(Tensor &lhs, const Tensor &rhs)
{
    // 创建 Div 实例并调用 forward_inplace
    Div op;
    op.forward_inplace(lhs, rhs);
}

}  // namespace functional

// 运算符重载放在 origin 命名空间下
Tensor operator/(const Tensor &lhs, const Tensor &rhs)
{
    return functional::div(lhs, rhs);
}

Tensor operator/(const Tensor &lhs, const Scalar &rhs)
{
    auto x = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    return functional::div(lhs, x);
}

Tensor operator/(const Scalar &lhs, const Tensor &rhs)
{
    auto x = Tensor(lhs, Shape({}), dtype(lhs.dtype()).device(rhs.device()));
    return functional::div(x, rhs);
}

// 就地操作运算符重载实现
Tensor &operator/=(Tensor &lhs, const Tensor &rhs)
{
    functional::div_(lhs, rhs);
    return lhs;
}

Tensor &operator/=(Tensor &lhs, const Scalar &rhs)
{
    auto temp = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    functional::div_(lhs, temp);
    return lhs;
}

}  // namespace origin
