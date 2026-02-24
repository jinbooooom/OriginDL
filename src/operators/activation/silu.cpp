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

    const Tensor &x = xs[0];

    // SiLU(x) = x * sigmoid(x)
    const Mat &x_mat = mat(x);
    auto y_mat       = x_mat.silu();
    auto y           = convert_mat_to_tensor(std::move(y_mat));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> SiLU::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("SiLU backward requires exactly 1 gradient, but got {}", gys.size());
    }

    const Tensor &gy = gys[0];
    const Tensor &x  = this->inputs_[0];

    // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    // 下沉到 Mat 层：当前 Mat 为 gy，传入前向输入 x
    const Mat &gy_mat = mat(gy);
    const Mat &x_mat  = mat(x);
    auto gx_mat       = gy_mat.silu_backward(x_mat);
    auto gx           = convert_mat_to_tensor(std::move(gx_mat));
    return std::vector<Tensor>{std::move(gx)};
}

Tensor silu(const Tensor &x)
{
    auto op = std::make_shared<SiLU>();
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
