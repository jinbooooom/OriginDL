#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Pow::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        throw std::runtime_error("Pow requires exactly 1 input");
    }
    auto x      = &mat(xs[0]);
    auto result = x->pow(exponent_);
    return std::vector<Tensor>{convert_mat_to_tensor(std::move(result))};
}

std::vector<Tensor> Pow::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("Pow backward requires exactly 1 gradient");
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

Tensor operator^(const Tensor &base, int exponent)
{
    return pow(base, exponent);
}

}  // namespace dl