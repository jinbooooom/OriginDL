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

    // exponent_ 在构造时已经提升为 float32 或 float64
    // 只需要将 base 提升到 exponent_ 的类型即可
    DataType promoted_dtype = exponent_.dtype();
    auto base_maybe         = TypePromotion::to_type_maybe_owned(xs[0], promoted_dtype);

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
    // exponent_ 在构造时已经提升为 float32 或 float64
    // inputs_[0] 是原始输入（由 Operator::setup_computation_graph 保存），需要提升到 exponent_ 的类型
    DataType promoted_dtype = exponent_.dtype();
    auto base_maybe         = TypePromotion::to_type_maybe_owned(this->inputs_[0], promoted_dtype);
    auto &x                 = mat(base_maybe);
    auto &gy                = mat(gys[0]);

    // ∂y/∂x = exponent * x^(exponent-1) * gy
    // 使用 exponent_ 的类型进行指数计算
    Scalar exponent_minus_1 = (promoted_dtype == DataType::kFloat64)
                                   ? Scalar(exponent_.to_float64() - 1.0)
                                   : Scalar(exponent_.to_float32() - 1.0f);
    // 优化：最大化利用 tmp 内存，所有操作都在 tmp 上进行
    auto tmp = x.pow(exponent_minus_1);
    tmp->mul_inplace(gy);
    tmp->mul_inplace(mat(Tensor(exponent_, Shape({}), dtype(promoted_dtype).device(x.device()))));
    auto gx = convert_mat_to_tensor(std::move(tmp));

    return std::vector<Tensor>{std::move(gx)};
}

void Pow::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (unlikely(&input1 != &kNullTensor_))
    {
        THROW_INVALID_ARG("Pow is a unary operator, cannot accept two operands");
    }

    // exponent_ 在构造时已经提升为 float32 或 float64
    // 只需要将 input0 提升到 exponent_ 的类型即可
    DataType promoted_dtype = exponent_.dtype();

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
    if (unlikely(xs.empty()))
    {
        THROW_RUNTIME_ERROR("pow requires at least 1 input tensor");
    }

    // 在构造 Pow 算子时进行类型提升，确保 exponent 只能是 float32 或 float64
    // 这样在 forward 和 backward 中就不需要重复计算类型提升了
    DataType base_dtype     = xs[0].dtype();
    DataType exponent_dtype = exponent.dtype();
    DataType promoted_dtype = TypePromotion::promote_types(base_dtype, exponent_dtype);

    // pow 操作的特殊规则：如果提升后的类型是整数，强制转换为 float32
    // 因为 pow 的结果可能是非整数，且 CUDA 层只支持浮点数计算
    // 这与 PyTorch 的行为一致：即使输入都是整数，pow 的结果也通常是浮点数
    if (promoted_dtype != DataType::kFloat32 && promoted_dtype != DataType::kFloat64)
    {
        promoted_dtype = DataType::kFloat32;
    }

    // 将 exponent 提升为 float32 或 float64
    Scalar promoted_exponent = (promoted_dtype == DataType::kFloat64) ? Scalar(exponent.to_float64())
                                                                      : Scalar(exponent.to_float32());

    auto op = std::make_shared<Pow>(promoted_exponent);
    return (*op)(xs)[0];
}

Tensor pow(const Tensor &base, const Scalar &exponent)
{
    return pow(std::vector<Tensor>{base}, exponent);
}

void pow_(Tensor &x, const Scalar &exponent)
{
    DataType base_dtype     = x.dtype();
    DataType exponent_dtype = exponent.dtype();
    DataType promoted_dtype = TypePromotion::promote_types(base_dtype, exponent_dtype);

    if (promoted_dtype != DataType::kFloat32 && promoted_dtype != DataType::kFloat64)
    {
        promoted_dtype = DataType::kFloat32;
    }

    Scalar promoted_exponent = (promoted_dtype == DataType::kFloat64) ? Scalar(exponent.to_float64())
                                                                      : Scalar(exponent.to_float32());
    Pow op(promoted_exponent);
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