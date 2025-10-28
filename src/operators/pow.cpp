#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/scalar.h"
#include "origin/utils/exception.h"
#include "origin/mat/type_promotion.h"

namespace origin
{

std::vector<Tensor> Pow::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Pow operator requires exactly 1 input, but got {}", xs.size());
    }

    // 使用统一的类型提升工具
    DataType base_dtype = xs[0].dtype();
    DataType exponent_dtype = exponent_.dtype();
    
    if (TypePromotion::needs_promotion(base_dtype, exponent_dtype))
    {
        // 自动类型提升
        DataType promoted_dtype = TypePromotion::promote_types(base_dtype, exponent_dtype);
        Tensor promoted_base = TypePromotion::to_type(xs[0], promoted_dtype);
        
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

    // 使用统一的类型提升工具
    DataType base_dtype = this->inputs_[0].dtype();
    DataType exponent_dtype = exponent_.dtype();
    
    if (TypePromotion::needs_promotion(base_dtype, exponent_dtype))
    {
        // 自动类型提升
        DataType promoted_dtype = TypePromotion::promote_types(base_dtype, exponent_dtype);
        Tensor promoted_base = TypePromotion::to_type(this->inputs_[0], promoted_dtype);
        
        // 使用提升后的base进行梯度计算
        auto &x = mat(promoted_base);
        auto &gy = mat(gys[0]);

        // ∂y/∂x = exponent * x^(exponent-1) * gy
        Scalar exponent_minus_1 = Scalar(exponent_.to_float32() - 1.0f);
        auto x_pow_minus_1 = x.pow(exponent_minus_1);
        auto temp_mult = *x_pow_minus_1 * gy;
        // 创建值为exponent的维度为0的张量
        auto scalar_exponent = Tensor(exponent_, Shape({}), dtype(promoted_dtype).device(x.device()));
        auto gx_result = mat(scalar_exponent) * *temp_mult;
        auto gx = convert_mat_to_tensor(std::move(gx_result));

        std::vector<Tensor> result;
        result.push_back(gx);
        return result;
    }

    // 类型匹配，直接计算
    auto &x = mat(this->inputs_[0]);
    auto &gy = mat(gys[0]);

    // ∂y/∂x = exponent * x^(exponent-1) * gy
    Scalar exponent_minus_1 = Scalar(exponent_.to_float32() - 1.0f);
    auto x_pow_minus_1 = x.pow(exponent_minus_1);
    auto temp_mult = *x_pow_minus_1 * gy;
    // 创建值为exponent的维度为0的张量
    auto scalar_exponent = Tensor(exponent_, Shape({}), dtype(x.dtype()).device(x.device()));
    auto gx_result = mat(scalar_exponent) * *temp_mult;
    auto gx = convert_mat_to_tensor(std::move(gx_result));

    std::vector<Tensor> result;
    result.push_back(gx);
    return result;
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

Tensor operator^(const Tensor &base, const Scalar &exponent)
{
    return pow(std::vector<Tensor>{base}, exponent);
}

}  // namespace origin