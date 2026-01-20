#include "origin/core/operator.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> SumTo::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("SumTo operator requires exactly 1 input, but got {}", xs.size());
    }
    auto result = mat(xs[0]).sum_to(this->shape_);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> SumTo::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("SumTo backward requires exactly 1 gradient, but got {}", gys.size());
    }
    auto x_shape = this->inputs_[0].shape();
    auto result  = mat(gys[0]).broadcast_to(x_shape);
    auto gx      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(gx)};
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

}  // namespace functional
}  // namespace origin