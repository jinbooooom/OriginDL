#include "base/dlUtils.h"
#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> SumTo::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1) {
        throw std::runtime_error("SumTo requires exactly 1 input");
    }
    auto x = xs[0].data();
    auto y = Tensor(utils::SumTo(x, this->shape_));
    return std::vector<Tensor>{y};
}

std::vector<Tensor> SumTo::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("SumTo backward requires exactly 1 gradient");
    }
    auto gy = gys[0].data();
    auto x = this->inputs_[0].data();
    auto gx = Tensor(utils::BroadcastTo(gy, x.dims()));
    return std::vector<Tensor>{gx};
}

Tensor sum_to(const std::vector<Tensor> &xs, const af::dim4 &shape)
{
    auto op = std::make_shared<SumTo>(shape);
    return (*op)(xs)[0];
}

Tensor sum_to(const Tensor &x, const af::dim4 &shape)
{
    std::vector<Tensor> inputs = {x};
    return sum_to(inputs, shape);
}

}  // namespace dl