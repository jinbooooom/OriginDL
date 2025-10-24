#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/origin/origin_mat.h"
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

    // 检查base张量与exponent标量的类型是否匹配，如果不匹配则进行类型提升
    DataType base_dtype = xs[0].dtype();
    DataType exponent_dtype = exponent_.dtype();
    
    if (base_dtype != exponent_dtype)
    {
        // 自动类型提升
        DataType promoted_dtype = promote_types(base_dtype, exponent_dtype);
        Tensor promoted_base = xs[0].dtype() == promoted_dtype ? xs[0] : xs[0].to(promoted_dtype);
        
        // 使用提升后的base进行运算
        auto &x = mat(promoted_base);
        auto result = x.pow(exponent_);
        return std::vector<Tensor>{convert_mat_to_tensor(std::move(result))};
    }

    // 类型匹配，直接运算
    auto &x = mat(xs[0]);
    auto result = x.pow(exponent_);
    return std::vector<Tensor>{convert_mat_to_tensor(std::move(result))};
}

std::vector<Tensor> Pow::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Pow backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // TODO: 未来需要在backward中也实现类型提升逻辑
    auto x  = &mat(this->inputs_[0]);
    auto gy = &mat(gys[0]);

    // ∂y/∂x = exponent * x^(exponent-1) * gy
    Scalar exponent_minus_1 = Scalar(exponent_.to_float32() - 1.0f);
    auto x_pow_minus_1 = x->pow(exponent_minus_1);
    auto temp_mult     = *x_pow_minus_1 * *gy;
    // 创建值为exponent的维度为0的张量
    auto scalar_exponent = Tensor(exponent_, Shape({}), dtype(x->dtype()).device(x->device()));
    auto gx_result  = mat(scalar_exponent) * *temp_mult;
    auto gx         = convert_mat_to_tensor(std::move(gx_result));

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
    auto op = std::make_shared<Pow>(exponent.to_float32());
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