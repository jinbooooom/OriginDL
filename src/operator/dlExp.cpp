#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Exp::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1) {
        throw std::runtime_error("Exp requires exactly 1 input");
    }
    auto x = xs[0].data();
    auto y = Tensor(af::exp(x));
    return std::vector<Tensor>{y};
}

std::vector<Tensor> Exp::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("Exp backward requires exactly 1 gradient");
    }
    auto x = this->inputs_[0].data();
    auto gy = gys[0].data();
    auto gx = Tensor(af::exp(x) * gy);
    return std::vector<Tensor>{gx};
}

Tensor exp(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Exp>();
    return (*op)(xs)[0];
}

Tensor exp(const Tensor &x)
{
    auto xs = std::vector<Tensor>();
    xs.emplace_back(x);
    return exp(xs);
}

}  // namespace dl
