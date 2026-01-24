#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/scalar.h"
#include "origin/mat/type_promotion.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Pow::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Pow operator requires exactly 1 input, but got {}", xs.size());
    }

    // 统一处理类型提升：Tensor 和 Scalar 之间的类型提升
    DataType base_dtype     = xs[0].dtype();
    DataType exponent_dtype = exponent_.dtype();
    DataType promoted_dtype = TypePromotion::promote_types(base_dtype, exponent_dtype);

    auto base_maybe = TypePromotion::to_type_maybe_owned(xs[0], promoted_dtype);

    // 使用提升后的base进行运算
    auto &x     = mat(base_maybe);
    auto result = x.pow(exponent_);
    return std::vector<Tensor>{convert_mat_to_tensor(std::move(result))};
}

std::vector<Tensor> Pow::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Pow backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // Pow的梯度计算：∂y/∂x = exponent * x^(exponent-1) * gy
    // 需要使用提升后的输入进行梯度计算
    DataType base_dtype     = this->inputs_[0].dtype();
    DataType exponent_dtype = exponent_.dtype();
    DataType promoted_dtype = TypePromotion::promote_types(base_dtype, exponent_dtype);

    auto base_maybe = TypePromotion::to_type_maybe_owned(this->inputs_[0], promoted_dtype);
    auto &x         = mat(base_maybe);
    auto &gy        = mat(gys[0]);

    // ∂y/∂x = exponent * x^(exponent-1) * gy
    Scalar exponent_minus_1 = Scalar(exponent_.to_float32() - 1.0f);
    auto x_pow_minus_1      = x.pow(exponent_minus_1);
    auto temp_mult          = *x_pow_minus_1 * gy;
    // 创建值为exponent的维度为0的张量
    auto scalar_exponent = Tensor(exponent_, Shape({}), dtype(promoted_dtype).device(x.device()));
    auto gx_result       = mat(scalar_exponent) * *temp_mult;
    auto gx              = convert_mat_to_tensor(std::move(gx_result));

    return std::vector<Tensor>{std::move(gx)};
}

void Pow::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (unlikely(&input1 != &kNullTensor_))
    {
        THROW_INVALID_ARG("Pow is a unary operator, cannot accept two operands");
    }

    // 统一处理类型提升：Tensor 和 Scalar 之间的类型提升
    DataType base_dtype     = input0.dtype();
    DataType exponent_dtype = exponent_.dtype();
    DataType promoted_dtype = TypePromotion::promote_types(base_dtype, exponent_dtype);

    // 因为 input0 需要原地修改，所以不用临时的 MaybeOwned<Tensor>，而是直接修改 input0
    if (input0.dtype() != promoted_dtype)
    {
        input0 = input0.to(promoted_dtype);
    }

    // 执行原地操作
    mat(input0).pow_inplace(exponent_);
}

// 支持Scalar类型的pow函数
Tensor pow(const std::vector<Tensor> &xs, const Scalar &exponent)
{
    // TODO:虽然cuda底层是用 float32 类型，但这里是否转换为float32还有待商榷
    auto op = std::make_shared<Pow>(exponent.to_float32());
    return (*op)(xs)[0];
}

Tensor pow(const Tensor &base, const Scalar &exponent)
{
    return pow(std::vector<Tensor>{base}, exponent);
}

void pow_(Tensor &x, const Scalar &exponent)
{
    // 创建 Pow 实例并调用 forward_inplace
    Pow op(exponent);
    op.forward_inplace(x, Operator::kNullTensor_);
}

}  // namespace functional

Tensor operator^(const Tensor &base, const Scalar &exponent)
{
    return functional::pow(std::vector<Tensor>{base}, exponent);
}

Tensor &operator^=(Tensor &x, const Scalar &exponent)
{
    functional::pow_(x, exponent);
    return x;
}

}  // namespace origin