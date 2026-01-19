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

    // 使用 MaybeOwned 优化版本：类型匹配时零开销
    auto [x0, x1] = TypePromotion::promote_tensors_maybe_owned(xs[0], xs[1]);
    auto result   = mat(x0) + mat(x1);
    auto y        = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Add::backward(const std::vector<Tensor> &gys)
{
    if (1 != gys.size())
    {
        THROW_RUNTIME_ERROR("Add backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // Add的梯度直接传递：gx0 = gy, gx1 = gy
    // 梯度类型已经和forward输出一致（提升后的类型），无需额外类型提升
    auto gx0 = gys[0];
    auto gx1 = gys[0];
    
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

void Add::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (&input1 == &kNullTensor_)
    {
        THROW_INVALID_ARG("Add requires two operands, cannot be used as unary operator");
    }

    // 原地操作：input0 = input0 + input1
    // 统一处理：无论是否需要类型提升，都使用相同的逻辑
    DataType promoted_type = TypePromotion::promote_types(input0.dtype(), input1.dtype());
    
    // 这里的处理和 forward 有细微的区别。
    // 在 forward 中不管是否需要转换，都直接使用 TypePromotion::promote_tensors_maybe_owned 生成两个 MaybeOwned<Tensor>，
    // 代码如下： auto [x0, x1] = TypePromotion::promote_tensors_maybe_owned(xs[0], xs[1]);
    // 而在 forward_inplace 中需要手动转换。
    // 因为 input0 需要原地修改，所以不用临时的 MaybeOwned<Tensor>，而是直接修改 input0。
    if (input0.dtype() != promoted_type)
    {
        input0 = input0.to(promoted_type);
    }
    
    // input1 使用 MaybeOwned 优化：类型匹配时借用，不匹配时创建新的 Tensor 并拥有所有权
    auto x1_maybe = TypePromotion::to_type_maybe_owned(input1, promoted_type);

    // 使用 mat() 方法获取 Mat 引用并执行原地操作
    mat(input0).add_inplace(mat(x1_maybe));
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
