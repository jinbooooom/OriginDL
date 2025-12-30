#include "origin/nn/layers/conv2d.h"
#include <cmath>
#include <vector>
#include "origin/core/operator.h"
#include "origin/mat/scalar.h"
#include "origin/operators/conv/conv2d.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace nn
{

Conv2d::Conv2d(int in_channels,
               int out_channels,
               std::pair<int, int> kernel_size,
               std::pair<int, int> stride,
               std::pair<int, int> pad,
               bool bias)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      pad_(pad),
      use_bias_(bias)
{
    // 验证参数有效性
    if (in_channels <= 0)
    {
        THROW_INVALID_ARG("Conv2d: in_channels must be positive, but got {}", in_channels);
    }
    if (out_channels <= 0)
    {
        THROW_INVALID_ARG("Conv2d: out_channels must be positive, but got {}", out_channels);
    }
    if (kernel_size.first <= 0 || kernel_size.second <= 0)
    {
        THROW_INVALID_ARG("Conv2d: kernel_size must be positive, but got ({}, {})", kernel_size.first,
                          kernel_size.second);
    }
    if (stride.first <= 0 || stride.second <= 0)
    {
        THROW_INVALID_ARG("Conv2d: stride must be positive, but got ({}, {})", stride.first, stride.second);
    }
    if (pad.first < 0 || pad.second < 0)
    {
        THROW_INVALID_ARG("Conv2d: pad must be non-negative, but got ({}, {})", pad.first, pad.second);
    }

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

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int pad, bool bias)
    : Conv2d(in_channels, out_channels, {kernel_size, kernel_size}, {stride, stride}, {pad, pad}, bias)
{}

Parameter Conv2d::init_weight()
{
    // Kaiming初始化（He初始化）：更适合ReLU激活函数
    // 对于卷积层，fan_in = in_channels * kernel_h * kernel_w
    int fan_in    = in_channels_ * kernel_size_.first * kernel_size_.second;
    float std_dev = std::sqrt(2.0f / static_cast<float>(fan_in));

    // 创建权重张量，形状 (out_channels, in_channels, kernel_h, kernel_w)
    auto weight_tensor =
        Tensor::randn(Shape{static_cast<size_t>(out_channels_), static_cast<size_t>(in_channels_),
                            static_cast<size_t>(kernel_size_.first), static_cast<size_t>(kernel_size_.second)},
                      TensorOptions(DataType::kFloat32));

    // 应用标准差缩放
    auto scaled_weight = weight_tensor * Scalar(std_dev);

    // 确保scaled_weight有正确的shape
    auto scaled_shape = scaled_weight.shape();
    if (scaled_shape.elements() == 0)
    {
        THROW_RUNTIME_ERROR("Conv2d init_weight: scaled_weight has empty shape!");
    }

    // 直接使用Parameter构造函数
    Parameter w(scaled_weight);

    // 验证Parameter的shape
    auto w_shape = w.shape();
    if (w_shape.elements() == 0)
    {
        THROW_RUNTIME_ERROR(
            "Conv2d init_weight: Parameter w has empty shape after construction! scaled_weight.shape() = {}",
            scaled_shape.to_string());
    }

    return w;
}

Parameter Conv2d::init_bias()
{
    if (use_bias_)
    {
        // 初始化偏置为零，形状 (out_channels,)
        auto bias_tensor = Tensor::zeros(Shape{static_cast<size_t>(out_channels_)}, TensorOptions(DataType::kFloat32));
        return Parameter(bias_tensor);
    }
    // 如果不使用偏置，返回一个默认的Parameter（不会使用）
    return Parameter();
}

void Conv2d::init_parameters()
{
    // 重新初始化（用于reset_parameters）
    weight_ = init_weight();
    if (use_bias_)
    {
        bias_ = init_bias();
    }
}

Tensor Conv2d::forward(const Tensor &input)
{
    // 检查weight_的状态
    auto w_shape = weight_.shape();
    if (w_shape.elements() == 0)
    {
        THROW_RUNTIME_ERROR("Conv2d weight is empty in forward! weight_.shape() = {}", w_shape.to_string());
    }

    // 验证输入形状
    auto input_shape = input.shape();
    if (input_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("Conv2d forward: input must be 4D (N, C, H, W), but got shape {}", input_shape.to_string());
    }

    // 验证输入通道数是否匹配
    if (input_shape[1] != static_cast<size_t>(in_channels_))
    {
        THROW_RUNTIME_ERROR(
            "Conv2d forward: input channel mismatch - input has {} channels, but layer expects {} channels",
            input_shape[1], in_channels_);
    }

    // 调用conv2d算子
    // 如果使用偏置，传递bias_的指针；否则传递nullptr
    const Tensor *bias_ptr = use_bias_ ? &static_cast<const Tensor &>(bias_) : nullptr;
    auto output            = conv2d(input, static_cast<const Tensor &>(weight_), bias_ptr, stride_, pad_);

    return output;
}

void Conv2d::reset_parameters()
{
    init_parameters();
}

}  // namespace nn
}  // namespace origin
