#include "origin/nn/layers/linear.h"
#include <cmath>
#include <vector>
#include "origin/core/operator.h"
#include "origin/mat/scalar.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace nn
{

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias)
{
    // 初始化参数（必须在成员变量初始化后才能调用）
    weight_ = init_weight();
    if (use_bias_)
    {
        bias_ = init_bias();
    }

    // 注册参数
    register_parameter("weight", weight_);
    if (use_bias_)
    {
        register_parameter("bias", bias_);
    }
}

Parameter Linear::init_weight()
{
    // 初始化权重（Xavier初始化：scale * randn）
    float scale        = std::sqrt(1.0f / static_cast<float>(in_features_));
    auto weight_tensor = Tensor::randn(Shape{static_cast<size_t>(in_features_), static_cast<size_t>(out_features_)},
                                       TensorOptions(DataType::kFloat32));
    // 应用scale
    auto scaled_weight = weight_tensor * Scalar(scale);

    // 确保scaled_weight有正确的shape
    auto scaled_shape = scaled_weight.shape();
    if (scaled_shape.elements() == 0)
    {
        THROW_RUNTIME_ERROR("init_weight: scaled_weight has empty shape!");
    }

    // 直接使用Parameter构造函数（显式创建Parameter对象）
    Parameter w(scaled_weight);

    // 验证Parameter的shape
    auto w_shape = w.shape();
    if (w_shape.elements() == 0)
    {
        THROW_RUNTIME_ERROR("init_weight: Parameter w has empty shape after construction! scaled_weight.shape() = {}",
                            scaled_shape.to_string());
    }

    return w;
}

Parameter Linear::init_bias()
{
    if (use_bias_)
    {
        // 初始化偏置为零
        // 注意：偏置形状设为 {1, out_features} 以方便与 {batch_size, out_features} 相加
        auto bias_tensor =
            Tensor::zeros(Shape{1, static_cast<size_t>(out_features_)}, TensorOptions(DataType::kFloat32));
        return Parameter(bias_tensor);
    }
    // 如果不使用偏置，返回一个默认的Parameter（不会使用）
    return Parameter();
}

void Linear::init_parameters()
{
    // 重新初始化（用于reset_parameters）
    weight_ = init_weight();
    if (use_bias_)
    {
        bias_ = init_bias();
    }
}

Tensor Linear::forward(const Tensor &input)
{
    // 矩阵乘法：y = input * weight
    // 检查weight_的状态
    auto w_shape = weight_.shape();
    if (w_shape.elements() == 0)
    {
        THROW_RUNTIME_ERROR("Weight is empty in forward! weight_.shape() = {}", w_shape.to_string());
    }

    // 直接使用weight_，因为Parameter继承自Tensor
    // 问题可能在于Parameter的拷贝/传递导致impl_丢失
    // 直接传递引用而不是拷贝
    auto output = functional::mat_mul(input, static_cast<const Tensor &>(weight_));

    // 添加偏置
    // 由于加法操作只支持相同形状或标量广播，需要先使用broadcast_to
    if (use_bias_)
    {
        // bias_ 的形状是 {1, out_features}，需要广播到 {batch_size, out_features}
        auto bias_broadcast = functional::broadcast_to(bias_, output.shape());
        output              = output + bias_broadcast;
    }

    return output;
}

void Linear::reset_parameters()
{
    init_parameters();
}

}  // namespace nn
}  // namespace origin
