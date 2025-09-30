#include "operator.h"

namespace origin
{

std::vector<Tensor> Transpose::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        throw std::runtime_error("Transpose requires exactly 1 input");
    }
    auto y = xs[0].transpose();
    return std::vector<Tensor>{y};
}

std::vector<Tensor> Transpose::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("Transpose backward requires exactly 1 gradient");
    }
    auto gx = gys[0].transpose();
    return std::vector<Tensor>{gx};
}

Tensor transpose(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Transpose>();
    return (*op)(xs)[0];
}

Tensor transpose(const Tensor &x)
{
    std::vector<Tensor> inputs = {x};
    return transpose(inputs);
}

}  // namespace origin