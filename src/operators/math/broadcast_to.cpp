#include "origin/core/operator.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> BroadcastTo::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("BroadcastTo operator requires exactly 1 input, but got {}", xs.size());
    }
    this->x_shape_ = xs[0].shape();
    auto result    = mat(xs[0]).broadcast_to(this->shape_);
    auto y         = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> BroadcastTo::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("BroadcastTo backward requires exactly 1 gradient, but got {}", gys.size());
    }
    auto result = mat(gys[0]).sum_to(this->x_shape_);
    auto gx     = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(gx)};
}

Tensor broadcast_to(const Tensor &x, const Shape &shape)
{
    auto op                    = std::make_shared<BroadcastTo>(shape);
    std::vector<Tensor> inputs = {x};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace functional
}  // namespace origin
