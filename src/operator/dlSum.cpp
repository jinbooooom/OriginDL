#include "base/dlUtils.h"
#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> Sum::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1) {
        throw std::runtime_error("Sum requires exactly 1 input");
    }
    auto x = xs[0].data();
    auto y = Tensor(af::sum(x, this->axis_));
    return std::vector<Tensor>{y};
}

std::vector<Tensor> Sum::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("Sum backward requires exactly 1 gradient");
    }
    auto gy = gys[0].data();
    auto x = this->inputs_[0].data();
    auto gx = Tensor(utils::BroadcastTo(gy, x.dims()));
    return std::vector<Tensor>{gx};
}

Tensor sum(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<Sum>(-1);  // -1 意味着所有元素相加
    return (*op)(xs)[0];
}

Tensor sum(const Tensor &x, int axis)
{
    auto op = std::make_shared<Sum>(axis);
    std::vector<Tensor> inputs = {x};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace dl