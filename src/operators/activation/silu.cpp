#include "origin/operators/activation/silu.h"
#include "origin/core/tensor.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> SiLU::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("SiLU operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x = xs[0];

    // SiLU(x) = x * sigmoid(x)
    auto sigmoid_x = sigmoid(x);
    auto result    = x * sigmoid_x;
    return std::vector<Tensor>{std::move(result)};
}

std::vector<Tensor> SiLU::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("SiLU backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x  = this->inputs_[0];

    // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    auto sigmoid_x         = sigmoid(x);
    auto one               = Tensor::ones(x.shape(), TensorOptions(x.dtype()).device(x.device()));
    auto one_minus_sigmoid = one - sigmoid_x;
    auto gx                = sigmoid_x * (one + x * one_minus_sigmoid);
    gx                     = gx * gy;
    return std::vector<Tensor>{std::move(gx)};
}

Tensor silu(const Tensor &x)
{
    auto op = std::make_shared<SiLU>();
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
