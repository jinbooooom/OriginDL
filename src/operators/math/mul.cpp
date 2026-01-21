#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/type_promotion.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Mul::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("Mul operator requires exactly 2 inputs, but got {}", xs.size());
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();

    auto [x0, x1] = TypePromotion::promote_tensors_maybe_owned(xs[0], xs[1]);
    auto result   = mat(x0) * mat(x1);
    auto y        = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Mul::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Mul backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // Mul的梯度：gx0 = gy * x1, gx1 = gy * x0
    // 需要使用提升后的输入进行梯度计算
    auto [x0, x1] = TypePromotion::promote_tensors_maybe_owned(this->inputs_[0], this->inputs_[1]);
    auto &gy      = mat(gys[0]);

    // 使用抽象层进行梯度计算
    auto gx0_result = gy * mat(x1);
    auto gx1_result = gy * mat(x0);

    auto gx0 = convert_mat_to_tensor(std::move(gx0_result));
    auto gx1 = convert_mat_to_tensor(std::move(gx1_result));

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

void Mul::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (unlikely(&input1 == &kNullTensor_))
    {
        THROW_INVALID_ARG("Mul requires two operands, cannot be used as unary operator");
    }

    // 原地操作：input0 = input0 * input1
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
    mat(input0).mul_inplace(mat(x1_maybe));
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

void mul_(Tensor &lhs, const Tensor &rhs)
{
    // 创建 Mul 实例并调用 forward_inplace
    Mul op;
    op.forward_inplace(lhs, rhs);
}

}  // namespace functional

// 运算符重载放在 origin 命名空间下
Tensor operator*(const Tensor &lhs, const Tensor &rhs)
{
    return functional::mul(lhs, rhs);
}

Tensor operator*(const Tensor &lhs, const Scalar &rhs)
{
    auto x = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    return functional::mul(lhs, x);
}

Tensor operator*(const Scalar &lhs, const Tensor &rhs)
{
    auto x = Tensor(lhs, Shape({}), dtype(lhs.dtype()).device(rhs.device()));
    return functional::mul(x, rhs);
}

// 就地操作运算符重载实现
Tensor &operator*=(Tensor &lhs, const Tensor &rhs)
{
    functional::mul_(lhs, rhs);
    return lhs;
}

Tensor &operator*=(Tensor &lhs, const Scalar &rhs)
{
    auto temp = Tensor(rhs, Shape({}), dtype(rhs.dtype()).device(lhs.device()));
    functional::mul_(lhs, temp);
    return lhs;
}

}  // namespace origin