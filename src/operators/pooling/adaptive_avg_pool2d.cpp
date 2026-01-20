#include "origin/operators/pooling/adaptive_avg_pool2d.h"
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/origin/origin_mat.h"
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
    const OriginMat &x_mat = static_cast<const OriginMat &>(mat(x));

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

    // 获取 Mat 引用并调用底层 adaptive_avg_pool2d_backward
    const OriginMat &gy_mat = static_cast<const OriginMat &>(mat(gy));
    const OriginMat &x_mat  = static_cast<const OriginMat &>(mat(x));

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
