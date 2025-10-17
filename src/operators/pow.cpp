#include "origin/core/operator.h"
#include "origin/mat/scalar.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> Pow::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Pow operator requires exactly 1 input, but got {}", xs.size());
    }
    auto x      = &mat(xs[0]);
    auto result = x->pow(exponent_);
    return std::vector<Tensor>{convert_mat_to_tensor(std::move(result))};
}

std::vector<Tensor> Pow::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Pow backward requires exactly 1 gradient, but got {}", gys.size());
    }
    auto x  = &mat(this->inputs_[0]);
    auto gy = &mat(gys[0]);

    // ∂y/∂x = exponent * x^(exponent-1) * gy
    auto x_pow_minus_1 = x->pow(exponent_.toDataT() - 1.0f);
    auto temp_mult     = *x_pow_minus_1 * *gy;
    auto gx_result     = *temp_mult * exponent_.toDataT();
    auto gx            = convert_mat_to_tensor(std::move(gx_result));

    return std::vector<Tensor>{gx};
}

Tensor pow(const std::vector<Tensor> &xs, data_t exponent)
{
    auto op = std::make_shared<Pow>(exponent);
    return (*op)(xs)[0];
}

// 支持Scalar类型的pow函数
Tensor pow(const std::vector<Tensor> &xs, const Scalar &exponent)
{
    auto op = std::make_shared<Pow>(exponent.toDataT());
    return (*op)(xs)[0];
}

// 支持标量指数的pow函数（使用模板，参考add.cpp的实现）
template <typename T>
Tensor pow(const Tensor &base, T exponent)
{
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);

    // 直接使用现有的pow函数，避免重复的PowHelper
    auto xs = std::vector<Tensor>();
    xs.emplace_back(base);
    return pow(xs, static_cast<data_t>(exponent));
}

template <typename T>
Tensor operator^(const Tensor &base, T exponent)
{
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    return pow(base, exponent);
}

// 模板实例化
template Tensor pow(const Tensor &base, float exponent);
template Tensor pow(const Tensor &base, double exponent);
template Tensor pow(const Tensor &base, int32_t exponent);
template Tensor pow(const Tensor &base, int8_t exponent);
template Tensor pow(const Tensor &base, unsigned long exponent);

template Tensor operator^(const Tensor &base, float exponent);
template Tensor operator^(const Tensor &base, double exponent);
template Tensor operator^(const Tensor &base, int32_t exponent);
template Tensor operator^(const Tensor &base, int8_t exponent);
template Tensor operator^(const Tensor &base, unsigned long exponent);

}  // namespace origin