#include "origin/nn/layers/rms_norm.h"
#include "origin/core/operator.h"
#include "origin/operators/normalization/rms_norm.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace nn
{

RMSNorm::RMSNorm(int normalized_shape, float eps) : normalized_shape_(normalized_shape), eps_(eps)
{
    if (unlikely(normalized_shape <= 0))
    {
        THROW_INVALID_ARG("RMSNorm: normalized_shape must be positive, but got {}", normalized_shape);
    }

    // 初始化 gamma 为全 1
    auto gamma_tensor = Tensor::ones(Shape{static_cast<size_t>(normalized_shape)},
                                     TensorOptions(DataType::kFloat32).requires_grad(true));
    gamma_            = Parameter(gamma_tensor);

    // 注册参数
    register_parameter("weight", gamma_);
}

Tensor RMSNorm::forward(const Tensor &input)
{
    // 验证输入形状：最后一个维度必须等于 normalized_shape_
    auto input_shape = input.shape();
    if (unlikely(input_shape.size() == 0))
    {
        THROW_RUNTIME_ERROR("RMSNorm forward: input must have at least 1 dimension, but got scalar");
    }

    size_t last_dim = input_shape[input_shape.size() - 1];
    if (unlikely(last_dim != static_cast<size_t>(normalized_shape_)))
    {
        THROW_RUNTIME_ERROR("RMSNorm forward: input last dimension {} does not match normalized_shape {}", last_dim,
                            normalized_shape_);
    }

    // 创建 RMSNorm Operator
    auto op = std::make_shared<functional::RMSNorm>(eps_);

    // 准备输入
    std::vector<Tensor> inputs;
    inputs.push_back(input);
    inputs.push_back(static_cast<const Tensor &>(gamma_));

    // 执行前向传播（使用 operator() 而不是直接调用 forward，以自动设置 requires_grad）
    auto outputs = (*op)(inputs);
    return outputs[0];
}

void RMSNorm::reset_parameters()
{
    // 重置 gamma 为全 1
    auto gamma_tensor = Tensor::ones(Shape{static_cast<size_t>(normalized_shape_)},
                                     TensorOptions(DataType::kFloat32).requires_grad(true));
    gamma_            = Parameter(gamma_tensor);
}

}  // namespace nn
}  // namespace origin
