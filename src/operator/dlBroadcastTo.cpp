#include "base/dlUtils.h"
#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> BroadcastTo::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1) {
        throw std::runtime_error("BroadcastTo requires exactly 1 input");
    }
    this->x_shape_ = xs[0].data().dims();
    auto y = Tensor(utils::BroadcastTo(xs[0].data(), this->shape_));
    return std::vector<Tensor>{y};
}

std::vector<Tensor> BroadcastTo::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("BroadcastTo backward requires exactly 1 gradient");
    }
    auto gx = Tensor(utils::SumTo(gys[0].data(), this->x_shape_));
    return std::vector<Tensor>{gx};
}

Tensor broadcast_to(const Tensor &x, const af::dim4 &shape)
{
    auto op = std::make_shared<BroadcastTo>(shape);
    std::vector<Tensor> inputs = {x};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace dl
