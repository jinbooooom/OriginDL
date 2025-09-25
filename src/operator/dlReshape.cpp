#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Reshape::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1) {
        throw std::runtime_error("Reshape requires exactly 1 input");
    }
    auto y = xs[0].reshape(this->shape_);
    return std::vector<Tensor>{y};
}

std::vector<Tensor> Reshape::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("Reshape backward requires exactly 1 gradient");
    }
    auto x = this->inputs_[0].data();
    auto gx = Tensor(af::moddims(gys[0].data(), x.dims()));
    return std::vector<Tensor>{gx};
}

Tensor reshape(const Tensor &x, const af::dim4 shape)
{
    auto op = std::make_shared<Reshape>(shape);
    std::vector<Tensor> inputs = {x};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace dl