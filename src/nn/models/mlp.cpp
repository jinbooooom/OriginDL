#include "origin/nn/models/mlp.h"
#include "origin/core/operator.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace nn
{

MLP::MLP(const std::vector<int> &hidden_sizes, std::function<Tensor(const Tensor &)> activation)
    : activation_(activation)
{
    if (unlikely(hidden_sizes.size() < 2))
    {
        THROW_INVALID_ARG("MLP requires at least 2 layer sizes (input and output), but got {}", hidden_sizes.size());
    }

    // 创建线性层并注册为子模块
    // 同时保存到 layers_ vector 以保持顺序（用于 forward()）
    for (size_t i = 0; i < hidden_sizes.size() - 1; ++i)
    {
        int in_features  = hidden_sizes[i];
        int out_features = hidden_sizes[i + 1];
        auto layer       = std::make_unique<Linear>(in_features, out_features, true);  // 使用偏置

        // 保存原始指针到 layers_（用于 forward() 时保证顺序）
        Linear *layer_ptr = layer.get();
        layers_.push_back(layer_ptr);

        // 注册为子模块，这样 named_parameters() 和 state_dict() 可以正确工作
        std::string layer_name = "layer_" + std::to_string(i);
        register_module(layer_name, std::move(layer));
    }

    // 如果没有指定激活函数，默认使用relu
    if (!activation_)
    {
        activation_ = [](const Tensor &x) { return functional::relu(x); };
    }
}

Tensor MLP::forward(const Tensor &input)
{
    Tensor x = input;

    // 逐层前向传播（使用 layers_ 保证顺序）
    for (size_t i = 0; i < layers_.size(); ++i)
    {
        // 线性变换
        x = layers_[i]->forward(x);

        // 除了最后一层，都应用激活函数
        if (i < layers_.size() - 1)
        {
            x = activation_(x);
        }
    }

    return x;
}

std::vector<Parameter *> MLP::parameters()
{
    // 直接使用基类的 parameters()，因为子模块已通过 register_module 注册
    return Module::parameters();
}

void MLP::to(Device device)
{
    // 直接使用基类的 to()，因为子模块已通过 register_module 注册
    Module::to(device);
}

}  // namespace nn
}  // namespace origin
