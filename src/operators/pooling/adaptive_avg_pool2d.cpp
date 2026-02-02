#include "origin/operators/pooling/adaptive_avg_pool2d.h"
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> AdaptiveAvgPool2d::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("AdaptiveAvgPool2d operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto x_shape = x.shape();

    // 检查输入形状：应该是 (N, C, H, W)
    if (unlikely(x_shape.size() != 4))
    {
        THROW_RUNTIME_ERROR("AdaptiveAvgPool2d forward: x must be 4D (N, C, H, W), but got shape {}",
                            x_shape.to_string());
    }

    // 获取 Mat 引用并调用底层 adaptive_avg_pool2d
    const Mat &x_mat = mat(x);

    auto result = x_mat.adaptive_avg_pool2d(output_size_);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> AdaptiveAvgPool2d::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("AdaptiveAvgPool2d backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x  = this->inputs_[0];

    // 自适应平均池化反向传播原理：
    // 自适应平均池化的反向传播原理与普通平均池化相同，区别在于窗口大小是自适应的。
    // 前向传播中，输出是输入窗口内所有元素的平均值：
    //   out = (x₁ + x₂ + ... + xₙ) / n
    // 其中窗口大小 n 根据输入和输出大小自适应计算：n = kernel_h × kernel_w
    // 对于窗口内的每个元素 xᵢ，其对输出的偏导数是：
    //   ∂out/∂xᵢ = 1/n
    // 根据链式法则，输入梯度为：
    //   dx = dout * (1/n)
    // 因此，上游梯度 dout 被均匀地分配到输入窗口的每个位置，每个位置获得 dout/n 的梯度贡献。

    // 获取 Mat 引用并调用底层 adaptive_avg_pool2d_backward
    const Mat &gy_mat = mat(gy);
    const Mat &x_mat  = mat(x);

    auto gx = x_mat.adaptive_avg_pool2d_backward(gy_mat, output_size_);
    return std::vector<Tensor>{convert_mat_to_tensor(std::move(gx))};
}

Tensor adaptive_avg_pool2d(const Tensor &x, std::pair<int, int> output_size)
{
    auto op = std::make_shared<AdaptiveAvgPool2d>(output_size);
    return (*op)(x);
}

Tensor adaptive_avg_pool2d(const Tensor &x, int output_size)
{
    return adaptive_avg_pool2d(x, std::make_pair(output_size, output_size));
}

}  // namespace functional
}  // namespace origin
