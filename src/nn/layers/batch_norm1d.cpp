#include "origin/nn/layers/batch_norm1d.h"
#include <cmath>
#include "origin/core/operator.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace nn
{

BatchNorm1d::BatchNorm1d(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum)
{
    if (unlikely(num_features <= 0))
    {
        THROW_INVALID_ARG("BatchNorm1d: num_features must be positive, but got {}", num_features);
    }

    // 初始化 gamma 为全 1
    auto gamma_tensor = Tensor::ones(Shape{static_cast<size_t>(num_features)}, TensorOptions(DataType::kFloat32));
    gamma_            = Parameter(gamma_tensor);

    // 初始化 beta 为全 0
    auto beta_tensor = Tensor::zeros(Shape{static_cast<size_t>(num_features)}, TensorOptions(DataType::kFloat32));
    beta_            = Parameter(beta_tensor);

    // 初始化 running_mean 为全 0
    running_mean_ = Tensor::zeros(Shape{static_cast<size_t>(num_features)}, TensorOptions(DataType::kFloat32));

    // 初始化 running_var 为全 1
    running_var_ = Tensor::ones(Shape{static_cast<size_t>(num_features)}, TensorOptions(DataType::kFloat32));

    // 注册参数
    register_parameter("weight", gamma_);
    register_parameter("bias", beta_);
}

Tensor BatchNorm1d::forward(const Tensor &input)
{
    // 验证输入形状
    auto input_shape = input.shape();
    if (unlikely(input_shape.size() != 2))
    {
        THROW_RUNTIME_ERROR("BatchNorm1d forward: input must be 2D (N, C), but got shape {}", input_shape.to_string());
    }

    if (unlikely(input_shape[1] != static_cast<size_t>(num_features_)))
    {
        THROW_RUNTIME_ERROR("BatchNorm1d forward: input feature size {} does not match num_features {}", input_shape[1],
                            num_features_);
    }

    // 创建 BatchNorm Operator
    auto op = std::make_shared<functional::BatchNorm>(is_training(), eps_, momentum_, 2);  // 2 表示 2D 输入

    // 准备输入
    std::vector<Tensor> inputs;
    inputs.push_back(input);
    inputs.push_back(static_cast<const Tensor &>(gamma_));
    inputs.push_back(static_cast<const Tensor &>(beta_));
    inputs.push_back(running_mean_);
    inputs.push_back(running_var_);

    // 执行前向传播
    auto outputs = op->forward(inputs);
    auto output  = outputs[0];

    // 如果训练模式，更新 running_mean 和 running_var
    if (is_training() && outputs.size() >= 3)
    {
        // 从 operator 中获取当前 batch 的均值和方差
        auto current_mean = outputs[1];
        auto current_var  = outputs[2];

        // 更新 running_mean: running_mean = momentum * running_mean + (1 - momentum) * current_mean
        auto running_mean_data = running_mean_.to_vector<float>();
        auto running_var_data  = running_var_.to_vector<float>();
        auto current_mean_data = current_mean.to_vector<float>();
        auto current_var_data  = current_var.to_vector<float>();

        for (size_t i = 0; i < static_cast<size_t>(num_features_); ++i)
        {
            running_mean_data[i] = momentum_ * running_mean_data[i] + (1.0f - momentum_) * current_mean_data[i];
            running_var_data[i]  = momentum_ * running_var_data[i] + (1.0f - momentum_) * current_var_data[i];
        }

        running_mean_ =
            Tensor(running_mean_data, running_mean_.shape(), dtype(DataType::kFloat32).device(input.device()));
        running_var_ = Tensor(running_var_data, running_var_.shape(), dtype(DataType::kFloat32).device(input.device()));
    }

    return output;
}

void BatchNorm1d::reset_parameters()
{
    // 重置 gamma 为全 1
    auto gamma_tensor = Tensor::ones(Shape{static_cast<size_t>(num_features_)}, TensorOptions(DataType::kFloat32));
    gamma_            = Parameter(gamma_tensor);

    // 重置 beta 为全 0
    auto beta_tensor = Tensor::zeros(Shape{static_cast<size_t>(num_features_)}, TensorOptions(DataType::kFloat32));
    beta_            = Parameter(beta_tensor);

    // 重置 running_mean 为全 0
    running_mean_ = Tensor::zeros(Shape{static_cast<size_t>(num_features_)}, TensorOptions(DataType::kFloat32));

    // 重置 running_var 为全 1
    running_var_ = Tensor::ones(Shape{static_cast<size_t>(num_features_)}, TensorOptions(DataType::kFloat32));
}

}  // namespace nn
}  // namespace origin
