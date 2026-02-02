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

    // 计算并保存 mask = (x > 0)
    const Mat &x_mat = mat(xs[0]);
    auto mask_result = x_mat > Scalar(0.0f);
    mask_            = convert_mat_to_tensor(std::move(mask_result));

    // 通过 Mat 层调用 relu
    auto result = x_mat.relu();
    auto y      = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> ReLU::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("ReLU backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // ReLU 的梯度：gx = gy * mask，其中 mask = (x > 0 ? 1 : 0)
    // 直接使用 forward 中保存的 mask_
    auto &gy = gys[0];

    // gx = gy * mask_
    const Mat &gy_mat   = mat(gy);
    const Mat &mask_mat = mat(mask_);
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
