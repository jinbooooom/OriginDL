#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Pow::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1) {
        throw std::runtime_error("Pow requires exactly 1 input");
    }
    auto x = xs[0].data();
    auto result = af::pow(x, exponent_);
    return std::vector<Tensor>{Tensor(result)};
}

std::vector<Tensor> Pow::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("Pow backward requires exactly 1 gradient");
    }
    auto x = this->inputs_[0].data();
    auto gy = gys[0].data();
    auto gx = Tensor(exponent_ * af::pow(x, exponent_ - 1) * gy);
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