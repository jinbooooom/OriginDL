#include "base/dlUtils.h"
#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> BroadcastTo::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        throw std::runtime_error("BroadcastTo requires exactly 1 input");
    }
    this->x_shape_ = xs[0].shape();
    auto result    = xs[0].data().broadcast_to(this->shape_);
    auto y         = Tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> BroadcastTo::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        throw std::runtime_error("BroadcastTo backward requires exactly 1 gradient");
    }
    auto result = gys[0].data().sum_to(this->x_shape_);
    auto gx     = Tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor broadcast_to(const Tensor &x, const Shape &shape)
{
    auto op                    = std::make_shared<BroadcastTo>(shape);
    std::vector<Tensor> inputs = {x};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace dl
