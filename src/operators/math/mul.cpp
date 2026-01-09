#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/type_promotion.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Mul::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        THROW_RUNTIME_ERROR("Mul operator requires exactly 2 inputs, but got {}", xs.size());
    }

    shape0_ = xs[0].shape();
    shape1_ = xs[1].shape();

    // 检查类型是否匹配，如果不匹配则进行类型提升
    if (xs[0].dtype() != xs[1].dtype())
    {
        // 自动类型提升
        DataType promoted_type = promote_types_rule(xs[0].dtype(), xs[1].dtype());
        Tensor x0              = xs[0].dtype() == promoted_type ? xs[0] : xs[0].to(promoted_type);
        Tensor x1              = xs[1].dtype() == promoted_type ? xs[1] : xs[1].to(promoted_type);

        // 使用提升后的张量进行运算
        auto result = mat(x0) * mat(x1);
        auto y      = convert_mat_to_tensor(std::move(result));

        std::vector<Tensor> outputs;
        outputs.push_back(y);
        return outputs;
    }

    // 类型匹配，直接运算
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
        THROW_RUNTIME_ERROR("Mul backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // TODO: 未来需要在backward中也实现类型提升逻辑

    auto &x0 = mat(this->inputs_[0]);
    auto &x1 = mat(this->inputs_[1]);
    auto &gy = mat(gys[0]);

    // 使用抽象层进行梯度计算
    auto gx0_result = gy * x1;
    auto gx1_result = gy * x0;

    auto gx0 = convert_mat_to_tensor(std::move(gx0_result));
    auto gx1 = convert_mat_to_tensor(std::move(gx1_result));

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

    std::vector<Tensor> gxs;
    gxs.push_back(gx0);
    gxs.push_back(gx1);
    return gxs;
}

void Mul::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (&input1 == &kNullTensor_)
    {
        THROW_INVALID_ARG("Mul requires two operands, cannot be used as unary operator");
    }

    // 原地操作：input0 = input0 * input1
    if (TypePromotion::needs_promotion({input0, input1}))
    {
        auto promoted_tensors = TypePromotion::promote_tensors({input0, input1});
        if (input0.dtype() != promoted_tensors[0].dtype())
        {
            input0 = input0.to(promoted_tensors[0].dtype());
        }
        Tensor input1_promoted = (input1.dtype() != promoted_tensors[1].dtype()) ? input1.to(promoted_tensors[1].dtype()) : input1;
        
        // 使用 mat() 方法获取 Mat 引用并执行原地操作
        mat(input0).mul_inplace(mat(input1_promoted));
    }
    else
    {
        // 类型匹配，直接执行原地操作
        mat(input0).mul_inplace(mat(input1));
    }
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

}  // namespace origin