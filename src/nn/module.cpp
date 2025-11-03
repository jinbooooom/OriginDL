#include "origin/nn/module.h"
#include "origin/utils/exception.h"

namespace origin
{

Module::Module() : training_(true) {}

Tensor Module::operator()(const Tensor &input)
{
    return forward(input);
}

std::vector<Parameter *> Module::parameters()
{
    std::vector<Parameter *> params;

    // 收集当前模块的参数
    for (auto &[name, param] : parameters_)
    {
        params.push_back(param);
    }

    // 递归收集子模块的参数
    for (auto &[name, module] : modules_)
    {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }

    return params;
}

void Module::register_parameter(const std::string &name, Parameter &param)
{
    // 检查是否已存在
    if (parameters_.find(name) != parameters_.end())
    {
        THROW_RUNTIME_ERROR("Parameter '{}' already registered", name);
    }
    parameters_[name] = &param;
}

void Module::register_module(const std::string &name, std::unique_ptr<Module> module)
{
    // 检查是否已存在
    if (modules_.find(name) != modules_.end())
    {
        THROW_RUNTIME_ERROR("Module '{}' already registered", name);
    }
    modules_[name] = std::move(module);
}

void Module::train(bool mode)
{
    training_ = mode;

    // 递归设置子模块
    for (auto &[name, module] : modules_)
    {
        module->train(mode);
    }
}

void Module::eval()
{
    train(false);
}

void Module::to(Device device)
{
    // 迁移所有参数到指定设备
    for (auto &[name, param] : parameters_)
    {
        *param = Parameter(param->to(device));
    }

    // 递归迁移子模块
    for (auto &[name, module] : modules_)
    {
        module->to(device);
    }
}

void Module::to(const TensorOptions &options)
{
    // 迁移所有参数到指定选项
    for (auto &[name, param] : parameters_)
    {
        *param = Parameter(param->to(options));
    }

    // 递归迁移子模块
    for (auto &[name, module] : modules_)
    {
        module->to(options);
    }
}

void Module::zero_grad()
{
    // 清除所有参数的梯度
    for (auto &[name, param] : parameters_)
    {
        param->clear_grad();
    }

    // 递归清除子模块的梯度
    for (auto &[name, module] : modules_)
    {
        module->zero_grad();
    }
}

}  // namespace origin
