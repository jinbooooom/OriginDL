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
    // 根据 requires_grad 决定是否保存输入形状
    if (xs[0].requires_grad())
    {
        // 需要梯度计算：保存输入形状用于反向传播
        this->x_shape_ = xs[0].shape();
    }
    // 如果不需要梯度，不保存 x_shape_（虽然 Shape 很小，但为了一致性）

    auto result = mat(xs[0]).broadcast_to(this->shape_);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> BroadcastTo::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("BroadcastTo backward requires exactly 1 gradient, but got {}", gys.size());
    }
    // 如果 x_shape_ 未初始化（requires_grad=false），说明不需要梯度计算
    // 这种情况不应该发生（因为 requires_grad=false 时不会构建计算图）
    // 检查 x_shape_ 是否为空（通过检查 size() 是否为 0）
    if (unlikely(x_shape_.size() == 0))
    {
        THROW_RUNTIME_ERROR("BroadcastTo backward: x_shape_ is not initialized. This should not happen when requires_grad=true");
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
