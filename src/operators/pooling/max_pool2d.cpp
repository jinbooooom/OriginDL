#include "origin/operators/pooling/max_pool2d.h"
#include <vector>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/conv_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> MaxPool2d::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("MaxPool2d operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto x_shape = x.shape();

    // 检查输入形状
    if (unlikely(x_shape.size() != 4))
    {
        THROW_RUNTIME_ERROR("MaxPool2d forward: x must be 4D (N, C, H, W), but got shape {}", x_shape.to_string());
    }

    // 获取 Mat 引用并调用底层 max_pool2d
    const Mat &x_mat = mat(x);

    // 清空之前的索引
    indices_.clear();
    auto result = x_mat.max_pool2d(kernel_size_, stride_, pad_, indices_);

    // 保存索引形状
    indices_shape_ = x_shape;  // 保存输入形状，用于验证

    auto y = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> MaxPool2d::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("MaxPool2d backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x  = this->inputs_[0];

    // 验证索引是否存在
    if (unlikely(indices_.empty()))
    {
        THROW_RUNTIME_ERROR("MaxPool2d backward: indices not found. forward() must be called before backward()");
    }

    // 最大池化反向传播原理：
    // 前向传播中，输出是输入窗口内所有元素的最大值：
    //   out = max(x₁, x₂, ..., xₙ)
    // 对于窗口内的每个元素 xᵢ，其对输出的偏导数是：
    //   ∂out/∂xᵢ = 1  (如果 xᵢ 是最大值)
    //   ∂out/∂xᵢ = 0  (如果 xᵢ 不是最大值)
    // 根据链式法则，输入梯度为：
    //   dx = dout * (1 或 0)
    // 因此，上游梯度 dout 只传递给最大值位置，其他位置的梯度为 0。
    // 前向传播时保存的索引用于确定哪个位置是最大值。

    // 获取 Mat 引用并调用底层 max_pool2d_backward
    const Mat &gy_mat = mat(gy);
    const Mat &x_mat  = mat(x);

    auto gx = x_mat.max_pool2d_backward(gy_mat, kernel_size_, stride_, pad_, indices_);
    return std::vector<Tensor>{convert_mat_to_tensor(std::move(gx))};
}

Tensor max_pool2d(const Tensor &x, std::pair<int, int> kernel_size, std::pair<int, int> stride, std::pair<int, int> pad)
{
    auto op = std::make_shared<MaxPool2d>(kernel_size, stride, pad);
    return (*op)(x);
}

Tensor max_pool2d(const Tensor &x, int kernel_size, int stride, int pad)
{
    return max_pool2d(x, std::make_pair(kernel_size, kernel_size), std::make_pair(stride, stride),
                      std::make_pair(pad, pad));
}

}  // namespace functional
}  // namespace origin
