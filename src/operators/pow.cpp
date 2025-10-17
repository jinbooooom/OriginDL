#include "origin/core/operator.h"
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
    auto x_pow_minus_1 = x->pow(exponent_ - 1);
    auto temp_mult     = *x_pow_minus_1 * *gy;
    auto gx_result     = *temp_mult * exponent_;
    auto gx            = convert_mat_to_tensor(std::move(gx_result));

    return std::vector<Tensor>{gx};
}

Tensor pow(const std::vector<Tensor> &xs, int exponent)
{
    auto op = std::make_shared<Pow>(exponent);
    return (*op)(xs)[0];
}

Tensor pow(const Tensor &base, int exponent)
{
    auto xs = std::vector<Tensor>();
    xs.emplace_back(base);
    return pow(xs, exponent);
}

// 支持float指数的pow函数（使用辅助类来访问protected方法）
class PowFloatHelper : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override { return {}; }
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override { return {}; }

    static Tensor compute(const Tensor &base, float exponent)
    {
        PowFloatHelper helper;
        auto result = helper.mat(const_cast<Tensor &>(base)).pow(static_cast<data_t>(exponent));
        return helper.convert_mat_to_tensor(std::move(result));
    }
};

Tensor pow(const Tensor &base, float exponent)
{
    return PowFloatHelper::compute(base, exponent);
}

Tensor operator^(const Tensor &base, int exponent)
{
    return pow(base, exponent);
}

}  // namespace origin