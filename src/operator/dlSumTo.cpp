#include "base/dlUtils.h"
#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> SumTo::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        throw std::runtime_error("SumTo requires exactly 1 input");
    }
    auto result = xs[0].data().sum_to(this->shape_);
    auto y      = Tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> SumTo::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("SumTo backward requires exactly 1 gradient");
    }
    auto x_shape = this->inputs_[0].shape();
    auto result  = gys[0].data().broadcast_to(x_shape);
    auto gx      = Tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor sum_to(const std::vector<Tensor> &xs, const Shape &shape)
{
    auto op = std::make_shared<SumTo>(shape);
    return (*op)(xs)[0];
}

Tensor sum_to(const Tensor &x, const Shape &shape)
{
    std::vector<Tensor> inputs = {x};
    return sum_to(inputs, shape);
}

}  // namespace dl