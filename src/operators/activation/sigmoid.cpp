#include "origin/core/operator.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Sigmoid::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Sigmoid operator requires exactly 1 input, but got {}", xs.size());
    }

    const Mat &x_mat = mat(xs[0]);
    auto result     = x_mat.sigmoid();
    auto y          = convert_mat_to_tensor(std::move(result));

    // 根据 requires_grad 决定是否保存中间结果
    if (xs[0].requires_grad())
    {
        // 需要梯度计算：保存 sigmoid(x) 用于反向传播
        sigmoid_x_ = y;
    }

    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Sigmoid::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Sigmoid backward requires exactly 1 gradient, but got {}", gys.size());
    }

    const Mat &gy_mat = mat(gys[0]);
    const Mat &y_mat  = mat(sigmoid_x_);
    auto gx_result    = gy_mat.sigmoid_backward(y_mat);
    auto gx           = convert_mat_to_tensor(std::move(gx_result));
    return std::vector<Tensor>{std::move(gx)};
}

Tensor sigmoid(const Tensor &x)
{
    auto op = std::make_shared<Sigmoid>();
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
