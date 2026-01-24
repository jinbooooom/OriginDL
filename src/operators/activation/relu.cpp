#include "origin/core/operator.h"
#include "origin/mat/mat.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> ReLU::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("ReLU operator requires exactly 1 input, but got {}", xs.size());
    }

    // 通过 Mat 层调用 relu
    auto result = mat(xs[0]).relu();
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> ReLU::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("ReLU backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // ReLU 的梯度：gx = gy * (x > 0 ? 1 : 0)
    // 计算 mask = relu'(x) = (x > 0 ? 1 : 0)
    // 使用 relu 的特性：如果 x > 0，relu(x) = x；如果 x <= 0，relu(x) = 0
    // 所以如果 relu(x) > 0，说明 x > 0，mask = 1；否则 mask = 0
    auto &x  = this->inputs_[0];
    auto &gy = gys[0];

    // 计算 relu(x)
    auto relu_x_result = mat(x).relu();
    auto relu_x        = convert_mat_to_tensor(std::move(relu_x_result));

    // 计算 mask = (relu(x) > 0 ? 1 : 0)
    // 使用近似：mask = relu(x) / (relu(x) + epsilon)
    // 当 relu(x) > 0 时，mask ≈ 1
    // 当 relu(x) = 0 时，mask = 0
    float epsilon               = 1e-8f;
    auto epsilon_tensor         = Tensor::full(relu_x.shape(), epsilon, dtype(relu_x.dtype()).device(relu_x.device()));
    const Mat &relu_x_mat       = mat(relu_x);
    const Mat &epsilon_mat      = mat(epsilon_tensor);
    auto relu_x_plus_eps_result = relu_x_mat + epsilon_mat;
    auto relu_x_plus_eps        = convert_mat_to_tensor(std::move(relu_x_plus_eps_result));

    const Mat &relu_x_plus_eps_mat = mat(relu_x_plus_eps);
    auto mask_result               = relu_x_mat / relu_x_plus_eps_mat;
    auto mask                      = convert_mat_to_tensor(std::move(mask_result));

    // gx = gy * mask
    const Mat &gy_mat   = mat(gy);
    const Mat &mask_mat = mat(mask);
    auto gx_result      = gy_mat * mask_mat;
    auto gx             = convert_mat_to_tensor(std::move(gx_result));
    return std::vector<Tensor>{std::move(gx)};
}

void ReLU::forward_inplace(Tensor &input0, const Tensor &input1)
{
    if (unlikely(&input1 != &kNullTensor_))
    {
        THROW_INVALID_ARG("ReLU is a unary operator, cannot accept two operands");
    }

    // 原地操作：input0 = relu(input0)
    mat(input0).relu_inplace();
}

Tensor relu(const Tensor &x)
{
    auto op = std::make_shared<ReLU>();
    return (*op)(x);
}

void relu_(Tensor &x)
{
    // 创建 ReLU 实例并调用 forward_inplace
    ReLU op;
    op.forward_inplace(x, Operator::kNullTensor_);
}

}  // namespace functional
}  // namespace origin
