#include "origin/operators/nn/upsample.h"
#include <cmath>
#include "origin/core/tensor.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> Upsample::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Upsample operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto x_shape = x.shape();

    if (unlikely(x_shape.size() != 4))
    {
        THROW_RUNTIME_ERROR("Upsample forward: x must be 4D (N, C, H, W), but got shape {}", x_shape.to_string());
    }

    // 计算输出形状
    Shape output_shape = x_shape;
    if (size_.first > 0 && size_.second > 0)
    {
        // 使用指定的目标大小
        output_shape[2] = size_.first;   // H
        output_shape[3] = size_.second;  // W
    }
    else
    {
        // 使用缩放因子
        output_shape[2] = static_cast<int>(std::round(x_shape[2] * scale_factor_.first));   // H
        output_shape[3] = static_cast<int>(std::round(x_shape[3] * scale_factor_.second));  // W
    }

    // 计算缩放因子
    int scale_h = output_shape[2] / x_shape[2];
    int scale_w = output_shape[3] / x_shape[3];

    // 获取 Mat 引用
    const Mat &x_mat = mat(x);

    // 使用 Mat 接口的 upsample 方法
    std::unique_ptr<Mat> result = x_mat.upsample(output_shape, scale_h, scale_w);

    auto y = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Upsample::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Upsample backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x  = this->inputs_[0];

    // 下采样梯度：对每个输入像素，累加所有对应的输出梯度
    auto x_shape  = x.shape();
    auto gy_shape = gy.shape();

    int scale_h = gy_shape[2] / x_shape[2];
    int scale_w = gy_shape[3] / x_shape[3];

    // 获取 Mat 引用
    const Mat &gy_mat = mat(gy);

    // 使用 Mat 接口的 upsample_backward 方法
    std::unique_ptr<Mat> result = gy_mat.upsample_backward(gy_mat, x_shape, scale_h, scale_w);

    auto gx = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(gx)};
}

Tensor upsample(const Tensor &x, const std::string &mode, std::pair<float, float> scale_factor)
{
    auto op = std::make_shared<Upsample>(mode, scale_factor);
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
