#include <algorithm>
#include <cmath>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Softmax::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Softmax operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto x_shape = x.shape();

    // 确定计算 softmax 的轴
    int axis = axis_;
    if (axis < 0)
    {
        axis = static_cast<int>(x_shape.size()) + axis;
    }
    if (unlikely(axis < 0 || axis >= static_cast<int>(x_shape.size())))
    {
        THROW_INVALID_ARG("Invalid axis {} for softmax. Tensor has {} dimensions", axis_, x_shape.size());
    }

    // 数值稳定性：先找到最大值（沿指定轴）
    // 使用 mat 层的 max 操作
    const Mat &x_mat = mat(x);
    auto max_result        = x_mat.max(axis);
    auto max_tensor        = convert_mat_to_tensor(std::move(max_result));

    // 为了正确广播，需要先将 max_tensor reshape 为可以在指定维度广播的形状
    // 例如：对于形状 (2, 3)，max(axis=1) 返回 (2,)，需要 reshape 为 (2, 1) 再 broadcast
    auto x_dims = x_shape.dims();
    std::vector<size_t> max_shape_dims = x_dims;
    max_shape_dims[axis]               = 1;  // 在 axis 维度设为 1
    Shape max_shape(max_shape_dims);
    auto max_reshaped = reshape(max_tensor, max_shape);

    // 广播 max 到原始形状
    auto max_broadcast = broadcast_to(max_reshaped, x_shape);

    // x - max(x)
    auto x_sub_max = x - max_broadcast;

    // exp(x - max(x))
    auto exp_x = exp(x_sub_max);

    // sum(exp(x - max(x)), axis=-1)
    // 注意：这里使用计算后的 axis（已经转换为正数），而不是 axis_
    // sum 操作在 axis == -1 时会对所有元素求和，所以我们需要传递转换后的 axis
    auto sum_exp = sum(exp_x, axis);

    // 为了正确广播，需要先将 sum_exp reshape 为可以在最后一个维度广播的形状
    // 例如：对于形状 (2, 2)，sum(axis=1) 返回 (2,)，需要 reshape 为 (2, 1) 再 broadcast
    std::vector<size_t> sum_shape_dims = x_dims;
    sum_shape_dims[axis]               = 1;  // 在 axis 维度设为 1
    Shape sum_shape(sum_shape_dims);
    auto sum_reshaped = reshape(sum_exp, sum_shape);

    // 广播 sum 到原始形状
    auto sum_broadcast = broadcast_to(sum_reshaped, x_shape);

    // exp(x - max(x)) / sum(exp(x - max(x)))
    auto result = exp_x / sum_broadcast;
    return std::vector<Tensor>{std::move(result)};
}

std::vector<Tensor> Softmax::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Softmax backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // Softmax 的梯度计算
    // 如果 y = softmax(x)，那么 ∂y/∂x = y * (gy - sum(gy * y))
    auto &x  = this->inputs_[0];
    auto &gy = gys[0];

    // 重新计算 softmax 输出（使用保存的中间结果会更高效，但这里为了简单重新计算）
    auto y = softmax(x, axis_);

    // gy * y
    auto gy_times_y = gy * y;

    // sum(gy * y, axis=-1)
    // 需要转换 axis，与 forward 中保持一致
    auto x_shape = x.shape();
    auto x_dims  = x_shape.dims();
    int axis     = axis_;
    if (axis < 0)
    {
        axis = static_cast<int>(x_dims.size()) + axis;
    }
    auto sum_gy_y = sum(gy_times_y, axis);

    // 为了正确广播，需要先将 sum_gy_y reshape 为可以在最后一个维度广播的形状
    std::vector<size_t> sum_shape_dims = x_dims;
    sum_shape_dims[axis]               = 1;  // 在 axis 维度设为 1
    Shape sum_shape(sum_shape_dims);
    auto sum_gy_y_reshaped = reshape(sum_gy_y, sum_shape);

    // 广播 sum 到原始形状
    auto sum_broadcast = broadcast_to(sum_gy_y_reshaped, x.shape());

    // gy - sum(gy * y)
    auto gy_minus_sum = gy - sum_broadcast;

    // y * (gy - sum(gy * y))
    auto gx = y * gy_minus_sum;

    return std::vector<Tensor>{std::move(gx)};
}

Tensor softmax(const Tensor &x, int axis)
{
    auto op = std::make_shared<Softmax>(axis);
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
