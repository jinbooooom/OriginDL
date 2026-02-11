#include "origin/nn/sequential.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{

void Sequential::add(std::unique_ptr<Module> module)
{
    // 添加到向量（保存所有权）
    modules_.push_back(std::move(module));
}

Tensor Sequential::forward(const Tensor &input)
{
    Tensor output = input;

    // 依次通过所有模块
    for (auto &module : modules_)
    {
        output = module->forward(output);
    }

    return output;
}

std::vector<Parameter *> Sequential::parameters()
{
    std::vector<Parameter *> params;

    // Sequential 是容器模块，通常不直接注册参数, 参数通常由子模块（如 Linear、Conv2d）持有, Sequential
    // 只负责按顺序组织子模块. 考虑到未来 Sequential 可能直接注册参数，所以还是调用基类方法首先收集当前模块自己的参数。
    auto base_params = Module::parameters();
    params.insert(params.end(), base_params.begin(), base_params.end());

    // 然后递归收集所有子模块的参数
    for (auto &module : modules_)
    {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }

    return params;
}

void Sequential::to(Device device)
{
    // 首先迁移当前模块自己的参数（如果有的话）
    Module::to(device);

    // 然后递归迁移所有子模块的参数（Sequential的子模块存储在modules_ vector中）
    for (auto &module : modules_)
    {
        module->to(device);
    }
}

Module &Sequential::operator[](size_t index)
{
    if (unlikely(index >= modules_.size()))
    {
        THROW_RUNTIME_ERROR("Index {} out of range for Sequential with {} modules", index, modules_.size());
    }
    return *modules_[index];
}

const Module &Sequential::operator[](size_t index) const
{
    if (unlikely(index >= modules_.size()))
    {
        THROW_RUNTIME_ERROR("Index {} out of range for Sequential with {} modules", index, modules_.size());
    }
    return *modules_[index];
}

}  // namespace origin
