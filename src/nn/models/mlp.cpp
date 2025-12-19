#include "origin/nn/models/mlp.h"
#include "origin/core/operator.h"
#include "origin/utils/exception.h"

namespace origin
{

MLP::MLP(const std::vector<int> &hidden_sizes, 
         std::function<Tensor(const Tensor &)> activation)
    : activation_(activation)
{
    if (hidden_sizes.size() < 2)
    {
        THROW_INVALID_ARG("MLP requires at least 2 layer sizes (input and output), but got {}", hidden_sizes.size());
    }

    // 创建线性层
    for (size_t i = 0; i < hidden_sizes.size() - 1; ++i)
    {
        int in_features = hidden_sizes[i];
        int out_features = hidden_sizes[i + 1];
        auto layer = std::make_unique<Linear>(in_features, out_features, true);  // 使用偏置
        layers_.push_back(std::move(layer));
        
        // Linear层的参数已经通过register_parameter注册了
        // 由于Linear继承自Layer，Layer继承自Module，参数会自动被Module收集
        // 我们不需要额外注册，只需要确保layers_中的模块能被访问到
    }

    // 如果没有指定激活函数，默认使用relu
    if (!activation_)
    {
        activation_ = [](const Tensor &x) { return relu(x); };
    }
}

Tensor MLP::forward(const Tensor &input)
{
    Tensor x = input;
    
    // 逐层前向传播
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
    std::vector<Parameter *> params;

    // 首先收集当前模块自己的参数（如果有的话）
    auto base_params = Module::parameters();
    params.insert(params.end(), base_params.begin(), base_params.end());

    // 收集所有层的参数
    for (auto &layer : layers_)
    {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}

void MLP::to(Device device)
{
    // 首先迁移当前模块自己的参数（如果有的话）
    Module::to(device);

    // 迁移所有层到指定设备
    for (auto &layer : layers_)
    {
        layer->to(device);
    }
}

}  // namespace origin

