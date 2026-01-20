#include "origin/core/operator.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Reshape::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Reshape operator requires exactly 1 input, but got {}", xs.size());
    }
    auto y = xs[0].reshape(this->shape_);
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Reshape::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Reshape backward requires exactly 1 gradient, but got {}", gys.size());
    }
    auto x_shape = this->inputs_[0].shape();
    auto result  = mat(gys[0]).reshape(x_shape);
    auto gx      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(gx)};
}

Tensor reshape(const Tensor &x, const Shape &shape)
{
    auto op                    = std::make_shared<Reshape>(shape);
    std::vector<Tensor> inputs = {x};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace functional
}  // namespace origin