#include "origin/core/operator.h"
#include "origin/mat/mat.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{
std::vector<Tensor> LeakyReLU::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("LeakyReLU operator requires exactly 1 input, but got {}", xs.size());
    }

    const Mat &x_mat = mat(xs[0]);

    // 根据requires_grad 觉得是否保存mask
    if (xs[0].requires_grad())
    {
        // 需要梯度计算：保存 mask (x > 0) 用于反向传播
        auto zero_tensor = Tensor(0, Shape{}, dtype(xs[0].dtype()).device(xs[0].device()));
        auto mask_result = x_mat > mat(zero_tensor);
        mask_            = convert_mat_to_tensor(std::move(mask_result));
    }

    // alpha 也要转为 mat
    auto alpha_tensor    = Tensor(Scalar(alpha_), Shape({}), dtype(x_mat.dtype()).device(x_mat.device()));
    const Mat &alpha_mat = mat(alpha_tensor);

    // 通过 Mat 层调用relu
    auto result = x_mat.leaky_relu(alpha_mat);
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> LeakyReLU::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("LeakyReLU backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // LeakyReLU 的梯度：gx = gy * (mask * (1-alpha) + alpha)，其中 mask = (x > 0 ? 1 : 0)
    // 直接使用 forward 中保存的 mask_
    auto &gy = gys[0];

    const Mat &gy_mat   = mat(gy);
    const Mat &mask_mat = mat(mask_);

    // 当 mask=1 时梯度为 1，当 mask=0 时梯度为 alpha
    // 利用：mask * (1-alpha) + alpha = alpha + mask * (1-alpha)

    auto alpha_tensor    = Tensor::full(mask_.shape(), Scalar(alpha_), dtype(gy.dtype()).device(gy.device()));
    const Mat &alpha_mat = mat(alpha_tensor);

    // 创建标量 (1-alpha) 用于广播
    auto one_minus_alpha           = Tensor(Scalar(1.0f - alpha_), Shape{}, dtype(gy.dtype()).device(gy.device()));
    const Mat &one_minus_alpha_mat = mat(one_minus_alpha);

    // gradient = alpha + mask * (1-alpha)
    auto mask_scaled = mask_mat * one_minus_alpha_mat;  // 广播：标量(1-alpha) * mask张量
    auto gradient    = alpha_mat + *mask_scaled;
    auto gx_result   = gy_mat * *gradient;
    auto gx          = convert_mat_to_tensor(std::move(gx_result));
    return std::vector<Tensor>{std::move(gx)};
}

Tensor leaky_relu(const Tensor &x, float alpha)
{
    auto op = std::make_shared<LeakyReLU>(alpha);
    return (*op)(x);
}
}  // namespace functional
}  // namespace origin