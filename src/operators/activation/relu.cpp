#include "origin/core/operator.h"
#include "origin/utils/exception.h"
#include "origin/mat/mat.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/scalar.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> ReLU::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("ReLU operator requires exactly 1 input, but got {}", xs.size());
    }

    // 通过 Mat 层调用 relu
    auto result = mat(xs[0]).relu();
    auto y      = convert_mat_to_tensor(std::move(result));
    
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> ReLU::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("ReLU backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // ReLU 的梯度：gx = gy * (x > 0 ? 1 : 0)
    // 计算 mask = relu'(x) = (x > 0 ? 1 : 0)
    // 使用 relu 的特性：如果 x > 0，relu(x) = x；如果 x <= 0，relu(x) = 0
    // 所以如果 relu(x) > 0，说明 x > 0，mask = 1；否则 mask = 0
    auto &x = this->inputs_[0];
    auto &gy = gys[0];
    
    // 计算 relu(x)
    auto relu_x_result = mat(x).relu();
    auto relu_x = convert_mat_to_tensor(std::move(relu_x_result));
    
    // 计算 mask = (relu(x) > 0 ? 1 : 0)
    // 使用近似：mask = relu(x) / (relu(x) + epsilon)
    // 当 relu(x) > 0 时，mask ≈ 1
    // 当 relu(x) = 0 时，mask = 0
    float epsilon = 1e-8f;
    auto epsilon_tensor = Tensor::full(relu_x.shape(), epsilon, dtype(relu_x.dtype()).device(relu_x.device()));
    const OriginMat &relu_x_mat = static_cast<const OriginMat &>(mat(relu_x));
    const OriginMat &epsilon_mat = static_cast<const OriginMat &>(mat(epsilon_tensor));
    auto relu_x_plus_eps_result = relu_x_mat + epsilon_mat;
    auto relu_x_plus_eps = convert_mat_to_tensor(std::move(relu_x_plus_eps_result));
    
    const OriginMat &relu_x_plus_eps_mat = static_cast<const OriginMat &>(mat(relu_x_plus_eps));
    auto mask_result = relu_x_mat / relu_x_plus_eps_mat;
    auto mask = convert_mat_to_tensor(std::move(mask_result));
    
    // gx = gy * mask
    const OriginMat &gy_mat = static_cast<const OriginMat &>(mat(gy));
    const OriginMat &mask_mat = static_cast<const OriginMat &>(mat(mask));
    auto gx_result = gy_mat * mask_mat;
    auto gx = convert_mat_to_tensor(std::move(gx_result));
    
    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor relu(const Tensor &x)
{
    auto op = std::make_shared<ReLU>();
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
