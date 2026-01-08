#include "origin/nn/module.h"
#include <set>
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

Module::StateDict Module::state_dict() const
{
    StateDict state_dict;
    auto named_params = named_parameters("");
    for (auto &[name, param] : named_params)
    {
        // 将 Parameter 转换为 Tensor（Parameter 继承自 Tensor，可以直接转换）
        state_dict[name] = static_cast<const Tensor &>(*param);
    }
    return state_dict;
}

void Module::load_state_dict(const StateDict &state_dict, bool strict)
{
    auto named_params = named_parameters();
    std::set<std::string> loaded_keys;

    // 加载参数
    for (auto &[name, param] : named_params)
    {
        auto it = state_dict.find(name);
        if (it != state_dict.end())
        {
            // 检查形状是否匹配
            if (param->shape() != it->second.shape())
            {
                THROW_RUNTIME_ERROR("Shape mismatch for parameter '{}': expected {}, got {}", name,
                                    param->shape().to_string(), it->second.shape().to_string());
            }
            // 更新参数值
            *param = Parameter(it->second);
            loaded_keys.insert(name);
        }
        else if (strict)
        {
            THROW_RUNTIME_ERROR("Missing parameter '{}' in state_dict (strict mode)", name);
        }
    }

    // 检查是否有未使用的键
    if (strict)
    {
        for (const auto &[key, value] : state_dict)
        {
            if (loaded_keys.find(key) == loaded_keys.end())
            {
                THROW_RUNTIME_ERROR("Unexpected parameter '{}' in state_dict (strict mode)", key);
            }
        }
    }
}

std::unordered_map<std::string, Parameter *> Module::named_parameters(const std::string &prefix)
{
    std::unordered_map<std::string, Parameter *> named_params;

    // 收集当前模块的参数
    for (auto &[name, param] : parameters_)
    {
        std::string full_name   = prefix.empty() ? name : prefix + "." + name;
        named_params[full_name] = param;
    }

    // 递归收集子模块的参数
    for (auto &[name, module] : modules_)
    {
        std::string module_prefix = prefix.empty() ? name : prefix + "." + name;
        auto sub_params           = module->named_parameters(module_prefix);
        named_params.insert(sub_params.begin(), sub_params.end());
    }

    return named_params;
}

std::unordered_map<std::string, const Parameter *> Module::named_parameters(const std::string &prefix) const
{
    std::unordered_map<std::string, const Parameter *> named_params;

    // 收集当前模块的参数
    for (const auto &[name, param] : parameters_)
    {
        std::string full_name   = prefix.empty() ? name : prefix + "." + name;
        named_params[full_name] = param;
    }

    // 递归收集子模块的参数
    for (const auto &[name, module] : modules_)
    {
        std::string module_prefix = prefix.empty() ? name : prefix + "." + name;
        auto sub_params           = module->named_parameters(module_prefix);
        named_params.insert(sub_params.begin(), sub_params.end());
    }

    return named_params;
}

void Module::load(const std::string &filepath, bool strict)
{
    auto state_dict = origin::load(filepath);
    load_state_dict(state_dict, strict);
}

}  // namespace origin
