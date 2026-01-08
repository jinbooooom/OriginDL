#include <algorithm>
#include "origin/core/operator.h"
#include "origin/utils/exception.h"
#include "origin/mat/mat.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/scalar.h"

namespace origin
{

std::vector<Tensor> ReLU::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("ReLU operator requires exactly 1 input, but got {}", xs.size());
    }

    // ReLU: y = max(0, x) = (x + |x|) / 2
    // 使用 mat() 接口，自动支持 GPU
    auto &x = xs[0];
    
    // 计算 x^2，使用 mat() 接口
    const OriginMat &x_mat = static_cast<const OriginMat &>(mat(x));
    auto x_squared_result = x_mat * x_mat;
    auto x_squared = convert_mat_to_tensor(std::move(x_squared_result));
    
    // 计算 sqrt(x^2) = |x|，使用 pow(x^2, 0.5) 替代 sqrt（支持 GPU）
    auto abs_x = pow(x_squared, Scalar(0.5f));
    
    // 计算 (x + |x|) / 2
    auto sum_result = x + abs_x;
    auto two = Tensor::full(x.shape(), 2.0f, dtype(x.dtype()).device(x.device()));
    auto result = sum_result / two;
    
    std::vector<Tensor> outputs;
    outputs.push_back(result);
    return outputs;
}

std::vector<Tensor> ReLU::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("ReLU backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // ReLU 的梯度：gx = gy * (x > 0 ? 1 : 0)
    // 使用 sign(x) 的近似：sign(x) ≈ (x + |x|) / (2 * |x| + epsilon)
    // 或者更简单：gx = gy * (x > 0) = gy * (sign(x) + 1) / 2
    // 但 sign(x) 在 x=0 时不确定，所以使用：gx = gy * (x + |x|) / (2 * |x| + epsilon)
    // 实际上，最简单的方式是：gx = gy * (x > 0)，但这需要比较操作
    
    // ReLU 的梯度：gx = gy * (x > 0 ? 1 : 0)
    // 使用近似：gx = gy * (x + sqrt(x^2)) / (2 * sqrt(x^2) + epsilon)
    // 当 x > 0 时，sqrt(x^2) = x，所以 (x + sqrt(x^2)) / (2 * sqrt(x^2) + epsilon) ≈ 1
    // 当 x < 0 时，sqrt(x^2) = -x，所以 (x + sqrt(x^2)) / (2 * sqrt(x^2) + epsilon) ≈ 0
    auto &x = this->inputs_[0];
    auto &gy = gys[0];
    
    // 计算 x^2 和 sqrt(x^2) = |x|，使用 pow(x^2, 0.5) 替代 sqrt（支持 GPU）
    const OriginMat &x_mat = static_cast<const OriginMat &>(mat(x));
    auto x_squared_result = x_mat * x_mat;
    auto x_squared = convert_mat_to_tensor(std::move(x_squared_result));
    auto abs_x = pow(x_squared, Scalar(0.5f));
    
    // 计算 (x + |x|)
    auto x_plus_abs = x + abs_x;
    
    // 计算 2 * |x| + epsilon (epsilon 很小，避免除零)
    float epsilon = 1e-8f;
    auto two_abs_plus_eps = abs_x * 2.0f + epsilon;
    
    // 计算 mask = (x + |x|) / (2 * |x| + epsilon)
    auto mask = x_plus_abs / two_abs_plus_eps;
    
    // gx = gy * mask
    auto gx = gy * mask;
    
    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor relu(const Tensor &x)
{
    auto op = std::make_shared<ReLU>();
    return (*op)(x);
}

}  // namespace origin
